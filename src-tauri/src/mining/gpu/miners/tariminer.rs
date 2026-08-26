// Copyright 2024. The Tari Project
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
// following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
// disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
// following disclaimer in the documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
// products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use axum::async_trait;
use log::{info, warn};
use regex::Regex;
use tari_shutdown::Shutdown;
use tokio::sync::watch::Sender;

#[cfg(target_os = "windows")]
use crate::utils::windows_setup_utils::add_firewall_rule;

use crate::{
    APPLICATION_FOLDER_ID, LOG_TARGET_APP_LOGIC, LOG_TARGET_STATUSES,
    configs::{
        config_mining::{ConfigMining, ConfigMiningContent},
        trait_config::ConfigImpl,
    },
    events_emitter::EventsEmitter,
    mining::{
        GpuConnectionType, MiningError,
        gpu::{
            consts::{GpuMinerStatus, GpuMinerType},
            interface::{GpuMinerInterfaceTrait, GpuMinerStatusInterface},
            manager::GpuManager,
            miners::GpuCommonInformation,
        },
    },
    process_adapter::{
        HandleUnhealthyResult, HealthStatus, ProcessAdapter, ProcessInstance, ProcessStartupSpec,
        StatusMonitor,
    },
    process_utils::launch_child_process,
};

/// TARI.Miner has no status API, it prints a periodic speed report to stdout instead.
/// A sample older than this is treated as if the miner stopped producing graphs.
const SPEED_SAMPLE_STALE_AFTER: Duration = Duration::from_secs(120);
/// Sub folder of the extracted release archive that holds the per architecture backends.
const BACKENDS_FOLDER: &str = "bin";

/// The CUDA backend that TARI.Miner ships one binary for per supported compute capability.
/// The upstream starter script picks the backend the same way, from `nvidia-smi --query-gpu=compute_cap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TariMinerBackend {
    /// RTX 30 series
    #[default]
    Sm86,
    /// RTX 40 series
    Sm89,
    /// RTX 50 series
    Sm120,
}

impl TariMinerBackend {
    fn from_compute_capability(compute_capability: &str) -> Option<Self> {
        match compute_capability.trim() {
            "8.6" => Some(TariMinerBackend::Sm86),
            "8.9" => Some(TariMinerBackend::Sm89),
            "12.0" => Some(TariMinerBackend::Sm120),
            _ => None,
        }
    }

    fn file_stem(self) -> &'static str {
        match self {
            TariMinerBackend::Sm86 => "tari_c29_pool_miner_sm_86",
            TariMinerBackend::Sm89 => "tari_c29_pool_miner_sm_89",
            TariMinerBackend::Sm120 => "tari_c29_pool_miner_sm_120",
        }
    }

    /// Path of the backend inside the extracted release archive.
    pub fn relative_binary_path(self) -> PathBuf {
        PathBuf::from(BACKENDS_FOLDER).join(self.file_stem())
    }
}

/// The archive ships every backend, but the binary resolver and the process watcher both need a
/// single path. Device detection publishes the backend that matches the GPU we are going to mine
/// with here so that both agree on which one to resolve, set permissions on and spawn.
static SELECTED_BACKEND: RwLock<TariMinerBackend> = RwLock::new(TariMinerBackend::Sm86);

/// Returns the backend that matches the GPU TARI.Miner will be started on.
/// Falls back to the default backend before any device was detected; every backend is present in
/// the archive, so resolving the binary path stays valid either way.
pub fn selected_backend() -> TariMinerBackend {
    SELECTED_BACKEND
        .read()
        .map(|backend| *backend)
        .unwrap_or_default()
}

fn publish_selected_backend(backend: TariMinerBackend) {
    match SELECTED_BACKEND.write() {
        Ok(mut selected) => *selected = backend,
        Err(error) => {
            warn!(target: LOG_TARGET_APP_LOGIC, "Could not store the selected TARI.Miner backend: {error}");
        }
    }
}

/// An NVIDIA device that TARI.Miner has a backend for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TariMinerDevice {
    pub common: GpuCommonInformation,
    pub backend: TariMinerBackend,
}

/// Parses `nvidia-smi --query-gpu=index,compute_cap,name --format=csv,noheader,nounits`.
/// Devices with a compute capability that has no backend in the release archive are dropped,
/// mirroring what the upstream starter script does.
fn parse_nvidia_smi_devices(output: &str) -> Vec<TariMinerDevice> {
    let mut devices = vec![];

    for line in output.lines() {
        let mut fields = line.split(',');
        let (Some(index), Some(compute_capability), Some(name)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };

        let Ok(device_id) = index.trim().parse::<u32>() else {
            continue;
        };

        let Some(backend) = TariMinerBackend::from_compute_capability(compute_capability) else {
            info!(
                target: LOG_TARGET_APP_LOGIC,
                "Skipping GPU {device_id}: compute capability {} is not supported by TARI.Miner",
                compute_capability.trim()
            );
            continue;
        };

        devices.push(TariMinerDevice {
            common: GpuCommonInformation {
                name: name.trim().to_string(),
                device_id,
            },
            backend,
        });
    }

    devices
}

/// TARI.Miner drives a single GPU per process, so we mine with the first device the user did not
/// exclude.
fn select_device<'a>(
    devices: &'a [TariMinerDevice],
    excluded_devices: &[u32],
) -> Result<&'a TariMinerDevice, anyhow::Error> {
    if devices.is_empty() {
        return Err(anyhow::anyhow!(
            "No TARI.Miner compatible GPU devices found"
        ));
    }

    devices
        .iter()
        .find(|device| !excluded_devices.contains(&device.common.device_id))
        .ok_or_else(|| MiningError::AllDevicesExcluded.into())
}

/// The manager hands us the pool specific worker name with its login separator prefixed
/// (`.Tari-universe` for LuckyPool, `/Tari-universe` for Kryptex). TARI.Miner takes the two apart,
/// so we split them back up here. A worker name without a leading separator keeps the miner's
/// own default.
fn split_worker_name(worker_name: &str) -> (Option<&str>, &str) {
    match worker_name.split_at_checked(1) {
        Some((separator @ ("." | "/"), worker)) => (Some(separator), worker),
        _ => (None, worker_name),
    }
}

/// Extracts the graph rate from TARI.Miner's periodic stdout report, which looks like
/// `speed 4.21 g/s | graphs=42 cycles=1 submitted=1 accepted=1 rejected=0`.
fn parse_speed_line(line: &str) -> Option<f64> {
    static SPEED_PATTERN: std::sync::LazyLock<Option<Regex>> =
        std::sync::LazyLock::new(|| Regex::new(r"speed[= ]([0-9]+(?:\.[0-9]+)?)\s*g/s").ok());

    let captures = SPEED_PATTERN.as_ref()?.captures(line)?;
    captures.get(1)?.as_str().parse::<f64>().ok()
}

/// Keeps the last graph rate TARI.Miner reported on stdout so the status monitor can read it.
#[derive(Clone, Default)]
pub struct TariMinerSpeedTracker {
    last_sample: Arc<Mutex<Option<(f64, Instant)>>>,
}

impl TariMinerSpeedTracker {
    fn record_line(&self, line: &str) {
        let Some(speed) = parse_speed_line(line) else {
            return;
        };

        match self.last_sample.lock() {
            Ok(mut sample) => *sample = Some((speed, Instant::now())),
            Err(error) => {
                warn!(target: LOG_TARGET_STATUSES, "Could not store the TARI.Miner speed sample: {error}");
            }
        }
    }

    /// Returns the last reported graph rate, or `None` when nothing was reported recently enough.
    fn latest_speed(&self) -> Option<f64> {
        let sample = self.last_sample.lock().ok()?;
        let (speed, reported_at) = (*sample)?;

        if reported_at.elapsed() > SPEED_SAMPLE_STALE_AFTER {
            return None;
        }

        Some(speed)
    }
}

#[derive(Default)]
pub struct TariMinerGpuMiner {
    pub tari_address: Option<String>,
    pub intensity_percentage: Option<u32>,
    pub worker_name: Option<String>,
    pub connection_type: Option<GpuConnectionType>,
    pub gpu_status_sender: Sender<GpuMinerStatus>,
    pub gpu_devices: Vec<TariMinerDevice>,
    pub excluded_devices: Vec<u32>,
}

impl TariMinerGpuMiner {
    pub fn new(gpu_status_sender: Sender<GpuMinerStatus>) -> Self {
        Self {
            tari_address: None,
            intensity_percentage: None,
            worker_name: None,
            connection_type: None,
            gpu_status_sender,
            gpu_devices: vec![],
            excluded_devices: vec![],
        }
    }

    /// Keeps the globally resolved backend in sync with the device we would mine with.
    fn refresh_selected_backend(&self) {
        if let Ok(device) = select_device(&self.gpu_devices, &self.excluded_devices) {
            publish_selected_backend(device.backend);
        }
    }
}

impl GpuMinerInterfaceTrait for TariMinerGpuMiner {
    async fn load_tari_address(&mut self, tari_address: &str) -> Result<(), anyhow::Error> {
        self.tari_address = Some(tari_address.to_string());
        Ok(())
    }
    async fn load_worker_name(&mut self, worker_name: Option<&str>) -> Result<(), anyhow::Error> {
        self.worker_name = worker_name.map(|name| name.to_string());
        Ok(())
    }
    async fn load_intensity_percentage(
        &mut self,
        intensity_percentage: u32,
    ) -> Result<(), anyhow::Error> {
        self.intensity_percentage = Some(intensity_percentage);
        Ok(())
    }
    async fn load_connection_type(
        &mut self,
        connection_type: GpuConnectionType,
    ) -> Result<(), anyhow::Error> {
        self.connection_type = Some(connection_type);
        Ok(())
    }

    async fn load_excluded_devices(
        &mut self,
        excluded_devices: Vec<u32>,
    ) -> Result<(), anyhow::Error> {
        self.excluded_devices = excluded_devices;
        self.refresh_selected_backend();
        Ok(())
    }

    /// TARI.Miner has no device listing of its own, it enumerates NVIDIA GPUs with `nvidia-smi`.
    async fn detect_devices(&mut self) -> Result<(), anyhow::Error> {
        let config_path =
            dirs::config_dir().ok_or_else(|| anyhow::anyhow!("Failed to get config directory"))?;

        let config_dir = config_path.join(APPLICATION_FOLDER_ID);

        let args = vec![
            "--query-gpu=index,compute_cap,name".to_string(),
            "--format=csv,noheader,nounits".to_string(),
        ];

        let result = launch_child_process(Path::new("nvidia-smi"), &config_dir, None, &args, true)
            .map_err(|e| {
                anyhow::anyhow!("Could not run nvidia-smi to enumerate NVIDIA GPUs: {e}")
            })?;

        let output = result.wait_with_output().await?;
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "nvidia-smi could not enumerate NVIDIA GPUs: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }

        let gpu_devices = parse_nvidia_smi_devices(&String::from_utf8_lossy(&output.stdout));
        if gpu_devices.is_empty() {
            return Err(anyhow::anyhow!(
                "No TARI.Miner compatible GPU devices found"
            ));
        }

        for device in &gpu_devices {
            info!(
                target: LOG_TARGET_APP_LOGIC,
                "TARI.Miner detected device {} [{}] -> {:?} backend",
                device.common.device_id, device.common.name, device.backend
            );
        }

        self.gpu_devices = gpu_devices;
        self.refresh_selected_backend();

        let common_devices: Vec<GpuCommonInformation> = self
            .gpu_devices
            .iter()
            .map(|device| device.common.clone())
            .collect();
        let devices_indexes: Vec<u32> = common_devices.iter().map(|d| d.device_id).collect();

        EventsEmitter::emit_detected_devices(common_devices).await;
        ConfigMining::update_field(
            ConfigMiningContent::populate_gpu_devices_settings,
            devices_indexes,
        )
        .await?;

        EventsEmitter::emit_update_gpu_devices_settings(
            ConfigMining::content().await.gpu_devices_settings().clone(),
        )
        .await;

        Ok(())
    }
}

impl ProcessAdapter for TariMinerGpuMiner {
    type ProcessInstance = ProcessInstance;
    type StatusMonitor = GpuMinerStatusInterface;

    fn spawn_inner(
        &self,
        base_folder: std::path::PathBuf,
        _config_folder: std::path::PathBuf,
        _log_folder: std::path::PathBuf,
        binary_version_path: std::path::PathBuf,
        _is_first_start: bool,
    ) -> Result<(Self::ProcessInstance, Self::StatusMonitor), anyhow::Error> {
        let inner_shutdown = Shutdown::new();
        let device = select_device(&self.gpu_devices, &self.excluded_devices)?;

        let mut args: Vec<String> =
            vec!["--device".to_string(), device.common.device_id.to_string()];

        if let Some(connection_type) = &self.connection_type {
            match connection_type {
                GpuConnectionType::Node {
                    node_grpc_address: _,
                } => {
                    return Err(anyhow::anyhow!("TARI.Miner does not support node mining"));
                }
                GpuConnectionType::Pool { pool_url } => {
                    args.push("--pool".to_string());
                    args.push(pool_url.clone());
                }
            }
        } else {
            return Err(anyhow::anyhow!(
                "Connection type must be set before starting the TariMinerGpuMiner"
            ));
        }

        if let Some(tari_address) = &self.tari_address {
            args.push("--wallet".to_string());
            args.push(tari_address.clone());
        } else {
            return Err(anyhow::anyhow!(
                "Tari address must be set before starting the TariMinerGpuMiner"
            ));
        }

        if let Some(worker_name) = &self.worker_name {
            let (login_separator, worker) = split_worker_name(worker_name);
            args.push("--worker".to_string());
            args.push(worker.to_string());
            if let Some(login_separator) = login_separator {
                args.push("--login-separator".to_string());
                args.push(login_separator.to_string());
            }
        }

        if let Some(intensity) = self.intensity_percentage {
            args.push("--intensity".to_string());
            args.push(intensity.clamp(1, 100).to_string());
        }

        info!(
            target: LOG_TARGET_APP_LOGIC,
            "TARI.Miner mining on device {} [{}] with binary: {}",
            device.common.device_id,
            device.common.name,
            binary_version_path.display()
        );

        #[cfg(target_os = "windows")]
        add_firewall_rule(
            format!("{}.exe", device.backend.file_stem()),
            binary_version_path.clone(),
        )?;

        let speed_tracker = TariMinerSpeedTracker::default();
        let output_sink = {
            let speed_tracker = speed_tracker.clone();
            Arc::new(move |line: &str| speed_tracker.record_line(line))
        };

        Ok((
            ProcessInstance {
                shutdown: inner_shutdown.clone(),
                startup_spec: ProcessStartupSpec {
                    file_path: binary_version_path,
                    envs: None,
                    args,
                    data_dir: base_folder,
                    pid_file_name: self.pid_file_name().to_string(),
                    name: self.name().to_string(),
                    output_sink: Some(output_sink),
                },
                handle: None,
            },
            GpuMinerStatusInterface::TariMiner(TariMinerGpuMinerStatusMonitor {
                gpu_status_sender: self.gpu_status_sender.clone(),
                speed_tracker,
            }),
        ))
    }

    fn name(&self) -> &str {
        "tariminer"
    }

    fn pid_file_name(&self) -> &str {
        "tariminer_pid"
    }
}

#[derive(Clone)]
pub struct TariMinerGpuMinerStatusMonitor {
    gpu_status_sender: Sender<GpuMinerStatus>,
    speed_tracker: TariMinerSpeedTracker,
}

// This is a flag to indicate if the fallback to other miner has already been triggered
// We want to avoid triggering it multiple times per session
static WAS_FALLBACK_TO_OTHER_MINER_TRIGGERED: AtomicBool = AtomicBool::new(false);

#[async_trait]
impl StatusMonitor for TariMinerGpuMinerStatusMonitor {
    async fn handle_unhealthy(
        &self,
        duration_since_last_healthy_status: Duration,
    ) -> Result<HandleUnhealthyResult, anyhow::Error> {
        info!(target: LOG_TARGET_STATUSES, "Handling unhealthy status for TariMinerGpuMiner | Duration since last healthy status: {:?}", duration_since_last_healthy_status.as_secs());
        if duration_since_last_healthy_status.as_secs().gt(&(60 * 3)) // Fallback after 3 minutes of unhealthiness
            && !WAS_FALLBACK_TO_OTHER_MINER_TRIGGERED.load(Ordering::SeqCst)
        {
            match GpuManager::write().await.handle_unhealthy_miner().await {
                Ok(_) => {
                    info!(target: LOG_TARGET_STATUSES, "TariMinerGpuMiner: fell back to another miner due to prolonged unhealthiness.");
                    WAS_FALLBACK_TO_OTHER_MINER_TRIGGERED.store(true, Ordering::SeqCst);
                    return Ok(HandleUnhealthyResult::Stop);
                }
                Err(error) => {
                    warn!(target: LOG_TARGET_STATUSES, "TariMinerGpuMiner: Failed to fall back to another miner: {error} | Continuing to monitor.");
                    return Ok(HandleUnhealthyResult::Continue);
                }
            }
        } else {
            return Ok(HandleUnhealthyResult::Continue);
        }
    }

    async fn check_health(&self, _uptime: Duration, _timeout_duration: Duration) -> HealthStatus {
        let status = self.status();
        let is_mining = status.is_mining;
        let _ = self.gpu_status_sender.send(status);

        if is_mining {
            if !GpuManager::read().await.is_current_miner_healthy().await {
                info!(target: LOG_TARGET_STATUSES, "Marking current miner as healthy again");
                let _unused = GpuManager::write().await.handle_healthy_miner().await;
            }
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        }
    }
}

impl TariMinerGpuMinerStatusMonitor {
    pub fn status(&self) -> GpuMinerStatus {
        match self.speed_tracker.latest_speed() {
            Some(hash_rate) => GpuMinerStatus {
                is_mining: true,
                hash_rate,
                estimated_earnings: 0,
                algorithm: GpuMinerType::TariMiner.main_algorithm(),
            },
            None => {
                GpuMinerStatus::default_with_algorithm(GpuMinerType::TariMiner.main_algorithm())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(device_id: u32, backend: TariMinerBackend) -> TariMinerDevice {
        TariMinerDevice {
            common: GpuCommonInformation {
                name: format!("GPU {device_id}"),
                device_id,
            },
            backend,
        }
    }

    #[test]
    fn backend_is_resolved_from_the_compute_capability() {
        assert_eq!(
            TariMinerBackend::from_compute_capability("8.6"),
            Some(TariMinerBackend::Sm86)
        );
        assert_eq!(
            TariMinerBackend::from_compute_capability(" 8.9 "),
            Some(TariMinerBackend::Sm89)
        );
        assert_eq!(
            TariMinerBackend::from_compute_capability("12.0"),
            Some(TariMinerBackend::Sm120)
        );
        assert_eq!(TariMinerBackend::from_compute_capability("7.5"), None);
    }

    #[test]
    fn nvidia_smi_output_is_parsed_and_unsupported_devices_are_dropped() {
        let output = "0, 8.9, NVIDIA GeForce RTX 4090\n1, 7.5, NVIDIA GeForce GTX 1660\n2, 12.0, NVIDIA GeForce RTX 5090\n";

        let devices = parse_nvidia_smi_devices(output);

        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].common.device_id, 0);
        assert_eq!(devices[0].common.name, "NVIDIA GeForce RTX 4090");
        assert_eq!(devices[0].backend, TariMinerBackend::Sm89);
        assert_eq!(devices[1].common.device_id, 2);
        assert_eq!(devices[1].backend, TariMinerBackend::Sm120);
    }

    #[test]
    fn malformed_nvidia_smi_lines_are_ignored() {
        let output = "\nnot a device row\n0, 8.6, NVIDIA GeForce RTX 3080\nx, 8.6, Broken index\n";

        let devices = parse_nvidia_smi_devices(output);

        assert_eq!(
            devices,
            vec![TariMinerDevice {
                common: GpuCommonInformation {
                    name: "NVIDIA GeForce RTX 3080".to_string(),
                    device_id: 0,
                },
                backend: TariMinerBackend::Sm86,
            }]
        );
    }

    #[test]
    fn the_first_device_that_is_not_excluded_is_selected() {
        let devices = vec![
            device(0, TariMinerBackend::Sm86),
            device(1, TariMinerBackend::Sm120),
        ];

        assert_eq!(select_device(&devices, &[]).ok(), Some(&devices[0]));
        assert_eq!(select_device(&devices, &[0]).ok(), Some(&devices[1]));
    }

    #[test]
    fn selecting_a_device_fails_when_all_of_them_are_excluded() {
        let devices = vec![device(0, TariMinerBackend::Sm86)];

        let error = select_device(&devices, &[0]).expect_err("all devices are excluded");

        assert!(matches!(
            error.downcast_ref::<MiningError>(),
            Some(MiningError::AllDevicesExcluded)
        ));
        assert!(select_device(&[], &[]).is_err());
    }

    #[test]
    fn the_login_separator_is_split_off_the_worker_name() {
        assert_eq!(
            split_worker_name(".Tari-universe"),
            (Some("."), "Tari-universe")
        );
        assert_eq!(
            split_worker_name("/Tari-universe"),
            (Some("/"), "Tari-universe")
        );
        assert_eq!(split_worker_name("Tari-universe"), (None, "Tari-universe"));
        assert_eq!(split_worker_name(""), (None, ""));
    }

    #[test]
    fn the_graph_rate_is_read_from_the_periodic_speed_report() {
        assert_eq!(
            parse_speed_line(
                "speed 4.21 g/s | graphs=42 cycles=1 submitted=1 accepted=1 rejected=0"
            ),
            Some(4.21)
        );
        assert_eq!(
            parse_speed_line(
                "graphs=42 elapsed=10.00s speed=4.215 g/s cycles=1 submitted=1 verify_failures=0"
            ),
            Some(4.215)
        );
        assert_eq!(parse_speed_line("share accepted (3 total)"), None);
    }

    #[test]
    fn a_recorded_speed_is_reported_and_a_missing_one_is_not() {
        let tracker = TariMinerSpeedTracker::default();
        assert_eq!(tracker.latest_speed(), None);

        tracker.record_line("share accepted (3 total)");
        assert_eq!(tracker.latest_speed(), None);

        tracker
            .record_line("speed 4.21 g/s | graphs=42 cycles=1 submitted=1 accepted=1 rejected=0");
        assert_eq!(tracker.latest_speed(), Some(4.21));
    }

    #[test]
    fn every_backend_resolves_to_a_binary_inside_the_archive() {
        assert_eq!(
            TariMinerBackend::Sm120.relative_binary_path(),
            PathBuf::from("bin").join("tari_c29_pool_miner_sm_120")
        );
    }
}

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

use super::trait_config::{ConfigContentImpl, ConfigImpl};
use crate::LOG_TARGET_APP_LOGIC;
use crate::events_emitter::EventsEmitter;
use crate::mining::gpu::consts::GpuMinerType;
use getset::{Getters, Setters};
use log::{info, warn};
use serde::{Deserialize, Deserializer, Serialize};
use std::time::Duration;
use std::{collections::HashMap, fmt::Display, sync::LazyLock, time::SystemTime};
use tauri::AppHandle;
use tokio::sync::RwLock;

pub const MINING_CONFIG_VERSION: u32 = 2;
static INSTANCE: LazyLock<RwLock<ConfigMining>> =
    LazyLock::new(|| RwLock::new(ConfigMining::new()));

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
pub enum MiningModeType {
    #[default]
    Eco,
    Turbo,
    Ludicrous,
    Custom,
    User,
}

impl Display for MiningModeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode_str = match self {
            MiningModeType::Eco => "Eco",
            MiningModeType::Turbo => "Turbo",
            MiningModeType::Ludicrous => "Ludicrous",
            MiningModeType::Custom => "Custom",
            MiningModeType::User => "User",
        };
        write!(f, "{mode_str}")
    }
}

impl From<&str> for MiningModeType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "eco" => MiningModeType::Eco,
            "turbo" => MiningModeType::Turbo,
            "ludicrous" => MiningModeType::Ludicrous,
            "custom" => MiningModeType::Custom,
            "user" => MiningModeType::User,
            _ => {
                warn!("Unknown mining mode type: {s}, defaulting to Eco");
                MiningModeType::Eco
            }
        }
    }
}

impl From<String> for MiningModeType {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MiningMode {
    pub mode_type: MiningModeType,
    pub mode_name: String,
    pub cpu_usage_percentage: u32,
    pub gpu_usage_percentage: u32,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct GpuDeviceSettings {
    device_id: u32,
    is_excluded: bool,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct GpuDevicesSettings(HashMap<u32, GpuDeviceSettings>);

impl GpuDevicesSettings {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn add(&mut self, device_id: u32) {
        self.0.entry(device_id).or_insert(GpuDeviceSettings {
            device_id,
            is_excluded: false,
        });
    }
    pub fn set_excluded(&mut self, device_id: u32, is_excluded: bool) {
        if let Some(settings) = self.0.get_mut(&device_id) {
            settings.is_excluded = is_excluded;
        }
    }
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PauseOnBatteryModeState {
    Enabled,
    Disabled,
    NotSupported,
}

impl PauseOnBatteryModeState {
    pub fn is_not_supported(&self) -> bool {
        matches!(self, PauseOnBatteryModeState::NotSupported)
    }
    pub fn is_enabled(&self) -> bool {
        matches!(self, PauseOnBatteryModeState::Enabled)
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(default)]
#[derive(Getters, Setters)]
#[getset(get = "pub", set = "pub")]
#[allow(clippy::struct_excessive_bools)]
pub struct ConfigMiningContent {
    version_counter: u32,
    created_at: SystemTime,
    selected_mining_mode: String,
    mining_modes: HashMap<String, MiningMode>,
    mine_on_app_start: bool,
    gpu_mining_enabled: bool,
    cpu_mining_enabled: bool,
    gpu_devices_settings: GpuDevicesSettings,
    /// The miner whose device numbering `gpu_devices_settings` is expressed in.
    /// Miners enumerate devices differently (lolMiner walks CUDA and OpenCL, TARI.Miner uses the
    /// nvidia-smi index), so a device id only means something together with the miner that produced
    /// it. `None` marks settings written before this field existed.
    #[serde(default, deserialize_with = "deserialize_optional_gpu_miner_type")]
    gpu_devices_settings_source: Option<GpuMinerType>,
    #[serde(default, deserialize_with = "deserialize_gpu_miner_type")]
    gpu_miner_type: GpuMinerType,
    squad_override: Option<String>,
    pause_on_battery_mode: PauseOnBatteryModeState,
    is_lolminer_tested: bool,
    is_gpu_mining_recommended: bool,

    eco_alert_needed: bool,
    mode_mining_times: HashMap<String, Duration>, // we only need Eco for now, but we can add to this if needed
}

impl Default for ConfigMiningContent {
    fn default() -> Self {
        Self {
            version_counter: MINING_CONFIG_VERSION,
            created_at: SystemTime::now(),
            selected_mining_mode: "Eco".to_string(),
            mine_on_app_start: true,
            mining_modes: HashMap::from([
                (
                    "Eco".to_string(),
                    MiningMode {
                        mode_type: MiningModeType::Eco,
                        mode_name: "Eco".to_string(),
                        cpu_usage_percentage: 1,
                        gpu_usage_percentage: 1,
                    },
                ),
                (
                    "Turbo".to_string(),
                    MiningMode {
                        mode_type: MiningModeType::Turbo,
                        mode_name: "Turbo".to_string(),
                        cpu_usage_percentage: 10,
                        gpu_usage_percentage: 10,
                    },
                ),
                (
                    "Ludicrous".to_string(),
                    MiningMode {
                        mode_type: MiningModeType::Ludicrous,
                        mode_name: "Ludicrous".to_string(),
                        cpu_usage_percentage: 85,
                        gpu_usage_percentage: 95,
                    },
                ),
                (
                    "Custom".to_string(),
                    MiningMode {
                        mode_type: MiningModeType::Custom,
                        mode_name: "Custom".to_string(),
                        cpu_usage_percentage: 75,
                        gpu_usage_percentage: 75,
                    },
                ),
            ]),
            gpu_mining_enabled: true,
            cpu_mining_enabled: true,
            gpu_devices_settings: GpuDevicesSettings::new(),
            gpu_devices_settings_source: None,
            gpu_miner_type: GpuMinerType::default(),
            pause_on_battery_mode: PauseOnBatteryModeState::Enabled,
            squad_override: None,
            is_lolminer_tested: false,
            is_gpu_mining_recommended: true,
            eco_alert_needed: true,
            mode_mining_times: HashMap::from([("Eco".to_string(), Duration::new(0, 0))]),
        }
    }
}
/// Tolerant deserializer for the selected GPU miner.
/// Configs written by older versions can still hold a miner that no longer exists (the SHA3 miners
/// that were removed), and those must not make the whole mining config fail to load.
fn deserialize_gpu_miner_type<'de, D>(deserializer: D) -> Result<GpuMinerType, D::Error>
where
    D: Deserializer<'de>,
{
    let raw_miner_type = String::deserialize(deserializer)?;
    Ok(GpuMinerType::from_name(&raw_miner_type).unwrap_or_else(|| {
        warn!(target: LOG_TARGET_APP_LOGIC, "Unknown gpu miner {raw_miner_type} in the mining config, falling back to the default one");
        GpuMinerType::default()
    }))
}

/// Tolerant deserializer for the miner that owns the stored device numbering.
/// Same reasoning as `deserialize_gpu_miner_type`: a miner name we no longer know means the stored
/// numbering is not ours, which is exactly what `None` says.
fn deserialize_optional_gpu_miner_type<'de, D>(
    deserializer: D,
) -> Result<Option<GpuMinerType>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw_miner_type = Option::<String>::deserialize(deserializer)?;
    Ok(raw_miner_type.and_then(|name| GpuMinerType::from_name(&name)))
}

impl ConfigContentImpl for ConfigMiningContent {}
impl ConfigMiningContent {
    pub fn update_custom_mode_cpu_usage(&mut self, cpu_usage_percentage: u32) -> &mut Self {
        if let Some(custom_mode) = self.mining_modes.get_mut("Custom") {
            custom_mode.cpu_usage_percentage = cpu_usage_percentage;
        }
        self
    }

    pub fn update_custom_mode_gpu_usage(&mut self, gpu_usage_percentage: u32) -> &mut Self {
        if let Some(custom_mode) = self.mining_modes.get_mut("Custom") {
            custom_mode.gpu_usage_percentage = gpu_usage_percentage;
        }
        self
    }

    /// Populate the GPU devices settings with the device IDs a miner detected.
    /// If a device ID already exists for that same miner, it will not be added again.
    ///
    /// Device ids are only comparable within one miner's enumeration, so settings left behind by a
    /// different miner are dropped rather than merged. Merging them would silently apply an
    /// exclusion to whichever device happens to share the id under the new miner, which can mean
    /// mining the wrong card or refusing to mine at all.
    pub fn populate_gpu_devices_settings(
        &mut self,
        (miner_type, device_ids): (GpuMinerType, Vec<u32>),
    ) -> &mut Self {
        if self.gpu_devices_settings_source.as_ref() != Some(&miner_type) {
            info!(
                target: LOG_TARGET_APP_LOGIC,
                "Gpu device settings were stored for {:?}, resetting them for {miner_type}",
                self.gpu_devices_settings_source
            );
            self.gpu_devices_settings = GpuDevicesSettings::new();
            self.gpu_devices_settings_source = Some(miner_type);
        }

        for device_id in device_ids {
            self.gpu_devices_settings.add(device_id);
        }

        self
    }

    pub fn enable_gpu_device_exclusion(&mut self, device_id: u32) -> &mut Self {
        self.gpu_devices_settings.set_excluded(device_id, true);
        self
    }

    pub fn disable_gpu_device_exclusion(&mut self, device_id: u32) -> &mut Self {
        self.gpu_devices_settings.set_excluded(device_id, false);
        self
    }

    pub fn get_selected_cpu_usage_percentage(&self) -> u32 {
        match self.mining_modes.get(&self.selected_mining_mode) {
            Some(mode) => mode.cpu_usage_percentage,
            None => {
                warn!("Mining mode '{}' not found", self.selected_mining_mode);
                0
            }
        }
    }

    pub fn get_excluded_devices(&self) -> Vec<u32> {
        self.gpu_devices_settings
            .0
            .iter()
            .filter_map(|(&device_id, settings)| {
                if settings.is_excluded {
                    Some(device_id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_selected_gpu_usage_percentage(&self) -> u32 {
        match self.mining_modes.get(&self.selected_mining_mode) {
            Some(mode) => mode.gpu_usage_percentage,
            None => {
                warn!("Mining mode '{}' not found", self.selected_mining_mode);
                0
            }
        }
    }
}
pub struct ConfigMining {
    content: ConfigMiningContent,
    app_handle: RwLock<Option<AppHandle>>,
}

impl ConfigMining {
    pub async fn initialize(app_handle: AppHandle) {
        let mut config = Self::current().write().await;
        config.load_app_handle(app_handle.clone()).await;
        drop(config);

        Self::_check_for_migration()
            .await
            .expect("Could not check for migration");
    }

    pub async fn update_mining_times(
        mode: MiningModeType,
        duration: u64,
    ) -> Result<(), anyhow::Error> {
        let mut mode_mining_times = Self::content().await.mode_mining_times;

        mode_mining_times
            .entry(mode.to_string())
            .and_modify(|t| *t = Duration::from_secs(t.as_secs() + duration));

        Self::update_field(
            ConfigMiningContent::set_mode_mining_times,
            mode_mining_times.clone(),
        )
        .await?;

        if mode.to_string() == "Eco" && Self::content().await.eco_alert_needed {
            let secs = mode_mining_times
                .get("Eco")
                .unwrap_or(&Duration::new(0, 0))
                .as_secs();
            let threshold = 3600 * 12; // 12 hours in seconds
            if secs >= threshold {
                EventsEmitter::emit_show_eco_alert().await;
            }
        }

        Ok(())
    }

    async fn _migrate() -> Result<(), anyhow::Error> {
        let current_version = Self::content().await.version_counter;

        // v0 -> v1 migration (existing mining modes migration)
        if current_version < 1 {
            let mut mining_modes = Self::content().await.mining_modes;
            let should_update_selected = !mining_modes.contains_key("Turbo")
                && Self::content().await.selected_mining_mode == "Eco";

            mining_modes
                .entry("Eco".to_string())
                .and_modify(|m| m.cpu_usage_percentage = 1)
                .and_modify(|m| m.gpu_usage_percentage = 1);

            let turbo = MiningMode {
                mode_type: MiningModeType::Turbo,
                mode_name: "Turbo".to_string(),
                cpu_usage_percentage: 10,
                gpu_usage_percentage: 10,
            };

            mining_modes
                .entry("Turbo".to_string())
                .or_insert_with(|| turbo);

            Self::update_field(ConfigMiningContent::set_mining_modes, mining_modes).await?;

            if should_update_selected {
                Self::update_field(
                    ConfigMiningContent::set_selected_mining_mode,
                    "Turbo".to_string(),
                )
                .await?;
            }
        }

        // v1 -> v2 migration (SHA3 removal)
        // Note: gpu_miner_type and gpu_engine fields will be ignored on next deserialize
        // since they're removed from the struct. GPU mining disable is handled at runtime
        // via is_supported_on_current_platform() check in phase_gpu_mining.rs

        Ok(())
    }

    async fn _check_for_migration() -> Result<(), anyhow::Error> {
        let current_version = Self::content().await.version_counter;
        if current_version < MINING_CONFIG_VERSION {
            info!(target: LOG_TARGET_APP_LOGIC, "Mining config needs migration v{current_version:?} => v{MINING_CONFIG_VERSION}");
            Self::_migrate().await?;
            Self::update_field(
                ConfigMiningContent::set_version_counter,
                MINING_CONFIG_VERSION,
            )
            .await?;
            return Ok(());
        }
        Ok(())
    }
}

impl ConfigImpl for ConfigMining {
    type Config = ConfigMiningContent;

    fn new() -> Self {
        Self {
            content: ConfigMining::_load_or_create(),
            app_handle: RwLock::new(None),
        }
    }

    fn current() -> &'static RwLock<Self> {
        &INSTANCE
    }

    async fn _get_app_handle(&self) -> Option<AppHandle> {
        self.app_handle.read().await.clone()
    }

    fn _get_name() -> String {
        "config_mining".to_string()
    }

    fn _get_content(&self) -> &Self::Config {
        &self.content
    }

    fn _get_content_mut(&mut self) -> &mut Self::Config {
        &mut self.content
    }

    async fn load_app_handle(&mut self, app_handle: AppHandle) {
        *self.app_handle.write().await = Some(app_handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_config_without_a_gpu_miner_keeps_the_default_one() {
        let content: ConfigMiningContent =
            serde_json::from_str(r#"{"selected_mining_mode":"Eco"}"#).expect("valid mining config");

        assert_eq!(*content.gpu_miner_type(), GpuMinerType::LolMiner);
        assert_eq!(content.selected_mining_mode(), "Eco");
    }

    #[test]
    fn a_saved_gpu_miner_is_restored() {
        let content: ConfigMiningContent =
            serde_json::from_str(r#"{"gpu_miner_type":"TariMiner"}"#).expect("valid mining config");

        assert_eq!(*content.gpu_miner_type(), GpuMinerType::TariMiner);
    }

    #[test]
    fn a_removed_gpu_miner_falls_back_to_the_default_one_instead_of_failing_the_load() {
        let content: ConfigMiningContent =
            serde_json::from_str(r#"{"gpu_miner_type":"Graxil","cpu_mining_enabled":false}"#)
                .expect("legacy mining configs still load");

        assert_eq!(*content.gpu_miner_type(), GpuMinerType::LolMiner);
        assert!(!content.cpu_mining_enabled());
    }

    #[test]
    fn device_settings_from_the_same_miner_are_kept() {
        let mut content = ConfigMiningContent::default();

        content.populate_gpu_devices_settings((GpuMinerType::LolMiner, vec![0, 1]));
        content.enable_gpu_device_exclusion(1);
        content.populate_gpu_devices_settings((GpuMinerType::LolMiner, vec![0, 1]));

        assert_eq!(content.get_excluded_devices(), vec![1]);
    }

    #[test]
    fn device_settings_are_dropped_when_another_miner_takes_over_the_numbering() {
        let mut content = ConfigMiningContent::default();

        // Under lolMiner's enumeration device 0 might be an AMD card the user turned off.
        content.populate_gpu_devices_settings((GpuMinerType::LolMiner, vec![0, 1]));
        content.enable_gpu_device_exclusion(0);
        assert_eq!(content.get_excluded_devices(), vec![0]);

        // TARI.Miner numbers from nvidia-smi, where device 0 is a completely different card, so
        // carrying the exclusion over would silently skip the card the user wants to mine with.
        content.populate_gpu_devices_settings((GpuMinerType::TariMiner, vec![0]));

        assert!(content.get_excluded_devices().is_empty());
        assert_eq!(
            *content.gpu_devices_settings_source(),
            Some(GpuMinerType::TariMiner)
        );
    }

    #[test]
    fn device_settings_written_before_the_source_was_tracked_are_dropped_once() {
        // Configs from before this field existed carry device ids with no miner attached, so the
        // first miner to detect devices has to start from a clean slate.
        let mut content: ConfigMiningContent = serde_json::from_str(
            r#"{"gpu_devices_settings":{"0":{"device_id":0,"is_excluded":true}}}"#,
        )
        .expect("valid mining config");
        assert_eq!(content.get_excluded_devices(), vec![0]);
        assert_eq!(*content.gpu_devices_settings_source(), None);

        content.populate_gpu_devices_settings((GpuMinerType::LolMiner, vec![0, 1]));

        assert!(content.get_excluded_devices().is_empty());
    }

    #[test]
    fn the_selected_gpu_miner_round_trips_through_serialization() {
        let mut content = ConfigMiningContent::default();
        content.set_gpu_miner_type(GpuMinerType::TariMiner);

        let serialized = serde_json::to_string(&content).expect("serializable mining config");
        let deserialized: ConfigMiningContent =
            serde_json::from_str(&serialized).expect("valid mining config");

        assert_eq!(*deserialized.gpu_miner_type(), GpuMinerType::TariMiner);
    }
}

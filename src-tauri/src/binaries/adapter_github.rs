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

use std::path::PathBuf;

use anyhow::{Error, anyhow};
use async_trait::async_trait;
use log::{error, info};
use regex::Regex;
use tari_common::configuration::Network;
use tokio::{fs::File, io::AsyncReadExt};

use crate::{
    APPLICATION_FOLDER_ID, LOG_TARGET_APP_LOGIC,
    requests::{
        clients::http_file_client::HttpFileClient, get_gh_download_url, get_mirror_download_url,
    },
};

use super::binaries_resolver::{BinaryDownloadInfo, LatestVersionApiAdapter};

/// Where a release publishes the checksum for an asset.
pub enum ChecksumSource {
    /// A `<asset>.sha256` sidecar next to the asset.
    PerAssetSidecar,
    /// One manifest for the whole release, listing `<hash>  <asset>` per line.
    SharedManifest(&'static str),
}

pub struct GithubReleasesAdapter {
    pub repo: String,
    pub owner: String,
    pub checksum_source: ChecksumSource,
}

/// Swaps the asset at the end of a release download url for another file in the same release.
fn sibling_release_file(asset_url: &str, file_name: &str) -> String {
    match asset_url.rsplit_once('/') {
        Some((release_url, _)) => format!("{release_url}/{file_name}"),
        None => file_name.to_string(),
    }
}

/// Pulls the hash for `asset_name` out of a checksum file.
/// Handles both the single line sidecar and a shared `<hash>  <asset>` manifest.
fn parse_expected_checksum(contents: &str, asset_name: &str) -> Result<String, Error> {
    let mut expected_hash = "";
    let regex = Regex::new(&format!(r"([a-f0-9]+)\s.{asset_name}"))
        .map_err(|e| anyhow!("Failed to create regex: {}", e))?;

    for line in contents.lines() {
        if let Some(caps) = regex.captures(line) {
            expected_hash = caps
                .get(1)
                .map(|hash| hash.as_str())
                .ok_or_else(|| anyhow!("Failed to extract hash from line: {}", line))?;
        }
    }
    Ok(expected_hash.to_string())
}

#[async_trait]
impl LatestVersionApiAdapter for GithubReleasesAdapter {
    async fn get_expected_checksum(
        &self,
        checksum_path: PathBuf,
        asset_name: &str,
    ) -> Result<String, Error> {
        let mut file_sha256 = File::open(checksum_path.clone()).await?;
        let mut buffer_sha256 = Vec::new();
        file_sha256.read_to_end(&mut buffer_sha256).await?;
        let contents =
            String::from_utf8(buffer_sha256).expect("Failed to read file contents as UTF-8");

        parse_expected_checksum(&contents, asset_name)
    }
    async fn download_and_get_checksum_path(
        &self,
        directory: PathBuf,
        download_info: BinaryDownloadInfo,
    ) -> Result<PathBuf, Error> {
        let (checksum_url, checksum_fallback_url) = match self.checksum_source {
            ChecksumSource::PerAssetSidecar => (
                format!("{}.sha256", download_info.main_url),
                format!("{}.sha256", download_info.fallback_url),
            ),
            ChecksumSource::SharedManifest(file_name) => (
                sibling_release_file(&download_info.main_url, file_name),
                sibling_release_file(&download_info.fallback_url, file_name),
            ),
        };

        match HttpFileClient::builder()
            .with_cloudflare_cache_check()
            .build(checksum_url.clone(), directory.clone())?
            .execute()
            .await
        {
            Ok(checksum_path) => Ok(checksum_path),
            Err(_) => {
                info!(target: LOG_TARGET_APP_LOGIC, "Fallback URL: {checksum_fallback_url}");
                HttpFileClient::builder()
                    .build(checksum_fallback_url.clone(), directory.clone())?
                    .execute()
                    .await
            }
        }
    }

    fn get_binary_folder(&self) -> Result<PathBuf, Error> {
        let cache_path =
            dirs::cache_dir().ok_or_else(|| anyhow::anyhow!("Failed to get cache directory"))?;

        let binary_folder_path = cache_path
            .join(APPLICATION_FOLDER_ID)
            .join("binaries")
            .join(&self.repo)
            .join(
                Network::get_current_or_user_setting_or_default()
                    .to_string()
                    .to_lowercase(),
            );

        if !binary_folder_path.exists() {
            std::fs::create_dir_all(&binary_folder_path).unwrap_or_else(|e| {
                error!(target: LOG_TARGET_APP_LOGIC, "Failed to create directory: {e}");
            });
        };

        Ok(binary_folder_path)
    }

    fn get_base_main_download_url(&self, version: &str) -> String {
        let base_url = get_mirror_download_url(&self.owner, &self.repo);
        format!("{base_url}/v{version}")
    }
    fn get_base_fallback_download_url(&self, version: &str) -> String {
        let base_url = get_gh_download_url(&self.owner, &self.repo);
        format!("{base_url}/v{version}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real manifest published with TARI.Miner v1.1.6.
    const SHARED_MANIFEST: &str = "\
b8a8839957b511582973438d8e9e4e52d4487f9ad30e2b1b462c71b8bbb21b12  TARI.Miner-v1.1.6-windows.zip
89a41e00182be21cdb7b1eceebcf0d5f43a6bc6e5a277c608e6ca7834dd193f8  TARI.Miner-v1.1.6-linux.tar.gz
9cac007c2c693f3007a251a1ac350a2831dcabc3b72bfd4c64361191e580dfbe  tari-miner-hiveos-1.1.6.tar.gz";

    #[test]
    fn the_hash_for_the_asset_we_downloaded_is_picked_out_of_a_shared_manifest() {
        assert_eq!(
            parse_expected_checksum(SHARED_MANIFEST, "TARI.Miner-v1.1.6-linux.tar.gz")
                .expect("parsed manifest"),
            "89a41e00182be21cdb7b1eceebcf0d5f43a6bc6e5a277c608e6ca7834dd193f8"
        );
        assert_eq!(
            parse_expected_checksum(SHARED_MANIFEST, "TARI.Miner-v1.1.6-windows.zip")
                .expect("parsed manifest"),
            "b8a8839957b511582973438d8e9e4e52d4487f9ad30e2b1b462c71b8bbb21b12"
        );
    }

    #[test]
    fn an_asset_missing_from_the_manifest_yields_no_hash_rather_than_another_assets() {
        assert_eq!(
            parse_expected_checksum(SHARED_MANIFEST, "TARI.Miner-v9.9.9-linux.tar.gz")
                .expect("parsed manifest"),
            ""
        );
    }

    #[test]
    fn a_single_line_sidecar_still_parses() {
        assert_eq!(
            parse_expected_checksum(
                "89a41e00182be21cdb7b1eceebcf0d5f43a6bc6e5a277c608e6ca7834dd193f8  asset.zip",
                "asset.zip"
            )
            .expect("parsed sidecar"),
            "89a41e00182be21cdb7b1eceebcf0d5f43a6bc6e5a277c608e6ca7834dd193f8"
        );
    }

    #[test]
    fn the_manifest_is_looked_for_next_to_the_asset_in_the_same_release() {
        assert_eq!(
            sibling_release_file(
                "https://cdn-universe.tari.com/tari-project/TARI.Miner/releases/download/v1.1.6/TARI.Miner-v1.1.6-linux.tar.gz",
                "SHA256SUMS.txt"
            ),
            "https://cdn-universe.tari.com/tari-project/TARI.Miner/releases/download/v1.1.6/SHA256SUMS.txt"
        );
    }
}

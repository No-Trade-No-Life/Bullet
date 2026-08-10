use std::{
    collections::BTreeSet,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

use chrono::NaiveTime;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct LiveConfig {
    pub account_id: String,
    pub bind_address: SocketAddr,
    pub history_tail_bars: usize,
    pub ctpd: CtpdConfig,
    pub remote_account: RemoteAccountConfig,
    pub instruments: Vec<InstrumentConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CtpdConfig {
    pub base_url: String,
    pub bearer_token_file: PathBuf,
    pub stale_after_ms: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RemoteAccountConfig {
    /// This has no serde default: deployment must explicitly choose its public
    /// access policy instead of silently exposing a simulated account.
    pub allow_unauthenticated: bool,
    pub bearer_token_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InstrumentConfig {
    /// Stable continuous-series name used by the historical lab, such as IF.
    pub symbol: String,
    /// Explicit currently tradable CTP contract. Roll this mapping at expiry.
    pub ctpd_instrument_id: String,
    pub parquet: PathBuf,
    pub target_contracts: i64,
    pub contract_multiplier: f64,
    /// Number of complete one-minute rows in the configured normal session.
    /// It makes the offline rule's required exit row knowable before entry.
    pub session_bar_count: usize,
    /// Latest Parquet `date` at which a signal can still have its twentieth
    /// following row in the configured trading session.
    pub last_executable_signal_time: String,
}

#[derive(Clone, Debug)]
pub struct Secrets {
    pub ctpd_bearer_token: String,
    pub remote_bearer_token: Option<String>,
}

impl LiveConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<(Self, Secrets), String> {
        let text =
            fs::read_to_string(path).map_err(|error| format!("cannot read config: {error}"))?;
        let config: Self =
            toml::from_str(&text).map_err(|error| format!("invalid config: {error}"))?;
        config.validate()?;
        let ctpd_bearer_token = read_secret(&config.ctpd.bearer_token_file)?;
        let remote_bearer_token =
            match (
                config.remote_account.allow_unauthenticated,
                config.remote_account.bearer_token_file.as_ref(),
            ) {
                (false, Some(path)) => Some(read_secret(path)?),
                (true, None) => None,
                (true, Some(_)) => {
                    return Err(
                        "remote_account cannot set a token when allow_unauthenticated=true".into(),
                    );
                }
                (false, None) => return Err(
                    "remote_account requires bearer_token_file when allow_unauthenticated=false"
                        .into(),
                ),
            };
        Ok((
            config,
            Secrets {
                ctpd_bearer_token,
                remote_bearer_token,
            },
        ))
    }

    fn validate(&self) -> Result<(), String> {
        if self.account_id.trim().is_empty() {
            return Err("account_id must not be empty".into());
        }
        if self.history_tail_bars == 0 {
            return Err("history_tail_bars must be positive".into());
        }
        if self.ctpd.base_url.trim().is_empty() || self.ctpd.stale_after_ms == 0 {
            return Err("ctpd base_url and stale_after_ms must be configured".into());
        }
        if self.instruments.is_empty() {
            return Err("at least one instrument is required".into());
        }
        let mut ids = BTreeSet::new();
        let mut symbols = BTreeSet::new();
        for instrument in &self.instruments {
            if instrument.symbol.trim().is_empty()
                || instrument.ctpd_instrument_id.trim().is_empty()
                || instrument.target_contracts <= 0
                || !instrument.contract_multiplier.is_finite()
                || instrument.contract_multiplier <= 0.0
                || instrument.session_bar_count < 80
                || NaiveTime::parse_from_str(&instrument.last_executable_signal_time, "%H:%M:%S")
                    .is_err()
            {
                return Err("each instrument requires names, positive contracts, a positive finite multiplier, at least 80 session bars, and an HH:MM:SS signal cutoff".into());
            }
            if !ids.insert(instrument.ctpd_instrument_id.clone()) {
                return Err("ctpd_instrument_id must be unique".into());
            }
            if !symbols.insert(instrument.symbol.clone()) {
                return Err("symbol must be unique so remote position IDs stay unique".into());
            }
            if self.history_tail_bars < instrument.session_bar_count {
                return Err(
                    "history_tail_bars must cover every bar in each configured session".into(),
                );
            }
        }
        Ok(())
    }
}

fn read_secret(path: &Path) -> Result<String, String> {
    let value = fs::read_to_string(path)
        .map_err(|error| format!("cannot read secret file {}: {error}", path.display()))?;
    let value = value.trim().to_owned();
    (!value.is_empty())
        .then_some(value)
        .ok_or_else(|| format!("secret file {} is empty", path.display()))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{CtpdConfig, InstrumentConfig, LiveConfig, RemoteAccountConfig};

    fn valid_config() -> LiveConfig {
        LiveConfig {
            account_id: "BULLET/lab0344-sim".into(),
            bind_address: "127.0.0.1:8091".parse().unwrap(),
            history_tail_bars: 240,
            ctpd: CtpdConfig {
                base_url: "http://127.0.0.1:8080".into(),
                bearer_token_file: PathBuf::from("/unused"),
                stale_after_ms: 10_000,
            },
            remote_account: RemoteAccountConfig {
                allow_unauthenticated: false,
                bearer_token_file: Some(PathBuf::from("/unused")),
            },
            instruments: vec![InstrumentConfig {
                symbol: "IF".into(),
                ctpd_instrument_id: "IF2609".into(),
                parquet: PathBuf::from("/unused"),
                target_contracts: 1,
                contract_multiplier: 300.0,
                session_bar_count: 240,
                last_executable_signal_time: "14:40:00".into(),
            }],
        }
    }

    #[test]
    fn requires_the_history_tail_to_cover_the_session() {
        let mut config = valid_config();
        config.history_tail_bars = 239;
        assert!(config.validate().unwrap_err().contains("history_tail_bars"));
    }

    #[test]
    fn rejects_duplicate_symbols_that_would_collide_remotely() {
        let mut config = valid_config();
        let mut duplicate = config.instruments[0].clone();
        duplicate.ctpd_instrument_id = "IF2612".into();
        config.instruments.push(duplicate);
        assert!(
            config
                .validate()
                .unwrap_err()
                .contains("symbol must be unique")
        );
    }
}

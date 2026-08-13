use std::{
    collections::BTreeSet,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

use chrono::NaiveTime;
use serde::Deserialize;

const LAB0334_SYMBOLS: [&str; 4] = ["IC8888", "IF8888", "IH8888", "IM8888"];

#[derive(Clone, Debug, Deserialize)]
pub struct LiveConfig {
    /// Stable remote-account identifier, not a human display name.
    pub account_id: String,
    pub bind_address: SocketAddr,
    /// Number of complete Parquet bars replayed before CTPD becomes authoritative.
    /// Set this above every configured file's row count to reconstruct the full
    /// causal label history used by the default lab-0334 arbitrator.
    pub history_seed_bars: usize,
    pub ctpd: CtpdConfig,
    pub remote_account: RemoteAccountConfig,
    pub linkit: Option<LinkitConfig>,
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
    /// This has no serde default: every deployment explicitly chooses whether
    /// its target account is publicly readable.
    pub allow_unauthenticated: bool,
    pub bearer_token_file: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LinkitConfig {
    /// Linkit Bot API origin. The token remains in a separate one-line file.
    pub base_url: String,
    pub bearer_token_file: PathBuf,
    /// Linkit username, not an Auth Mini user ID.
    pub recipient_username: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InstrumentConfig {
    /// Exact E-Works continuous-series identifier, for example `IF8888`.
    pub symbol: String,
    /// CTPD synthetic continuous index used for both SSE and Kline recovery,
    /// for example `IDX-CFFEX-IF`.
    pub market_instrument_id: String,
    /// Explicit front contract exposed as the simulated target position. It is
    /// a deployment-owned roll mapping and is never inferred from `8888`.
    pub target_instrument_id: String,
    pub parquet: PathBuf,
    /// The whole-portfolio contract amount for a position_weight of 1.0. A
    /// value such as 10 keeps the lab's 0.70/0.30 sleeves representable.
    pub full_weight_contracts: f64,
    pub contract_multiplier: f64,
    /// The expected number of completed one-minute bars in the normal CFFEX
    /// day session. lab0334's late-exit rule is expressed in remaining bars.
    pub session_bar_count: usize,
    /// The bar-end time whose open is the lab's EOD exit price. The current
    /// E-Works data contract is 15:00:00.
    pub session_end_time: String,
}

#[derive(Clone, Debug)]
pub struct Secrets {
    pub ctpd_bearer_token: String,
    pub remote_bearer_token: Option<String>,
    pub linkit_bearer_token: Option<String>,
}

impl LiveConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<(Self, Secrets), String> {
        let config = Self::load_without_secrets(path)?;
        let ctpd_bearer_token = read_secret(&config.ctpd.bearer_token_file)?;
        let remote_bearer_token = match (
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
            (false, None) => {
                return Err(
                    "remote_account requires bearer_token_file when allow_unauthenticated=false"
                        .into(),
                );
            }
        };
        let linkit_bearer_token = config
            .linkit
            .as_ref()
            .map(|linkit| read_secret(&linkit.bearer_token_file))
            .transpose()?;
        Ok((
            config,
            Secrets {
                ctpd_bearer_token,
                remote_bearer_token,
                linkit_bearer_token,
            },
        ))
    }

    /// Parses only public configuration and validates the Parquet seed. This
    /// supports bounded startup profiling without reading API credentials.
    pub fn load_without_secrets(path: impl AsRef<Path>) -> Result<Self, String> {
        let text =
            fs::read_to_string(path).map_err(|error| format!("cannot read config: {error}"))?;
        let config: Self =
            toml::from_str(&text).map_err(|error| format!("invalid config: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.account_id.trim().is_empty() {
            return Err("account_id must not be empty".into());
        }
        if self.history_seed_bars < 5_000 {
            return Err("history_seed_bars must be at least 5000 for lab0334 ATR history".into());
        }
        if self.ctpd.base_url.trim().is_empty() || self.ctpd.stale_after_ms == 0 {
            return Err("ctpd base_url and stale_after_ms must be configured".into());
        }
        if let Some(linkit) = &self.linkit
            && (linkit.base_url.trim().is_empty() || linkit.recipient_username.trim().is_empty())
        {
            return Err("linkit base_url and recipient_username must be configured".into());
        }
        if self.instruments.len() != LAB0334_SYMBOLS.len() {
            return Err("lab0334 requires exactly IC8888, IF8888, IH8888 and IM8888".into());
        }

        let mut symbols = BTreeSet::new();
        let mut market_ids = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for instrument in &self.instruments {
            let end_time = NaiveTime::parse_from_str(&instrument.session_end_time, "%H:%M:%S")
                .map_err(|_| "each session_end_time must be HH:MM:SS")?;
            if end_time != NaiveTime::from_hms_opt(15, 0, 0).expect("valid fixed close")
                || instrument.symbol.trim().is_empty()
                || instrument.market_instrument_id.trim().is_empty()
                || instrument.target_instrument_id.trim().is_empty()
                || !instrument.full_weight_contracts.is_finite()
                || instrument.full_weight_contracts <= 0.0
                || !instrument.contract_multiplier.is_finite()
                || instrument.contract_multiplier <= 0.0
                || instrument.session_bar_count != 240
            {
                return Err("each lab0334 instrument requires the 240-bar 15:00:00 session, names, and positive finite contract sizing".into());
            }
            symbols.insert(instrument.symbol.as_str());
            if !market_ids.insert(instrument.market_instrument_id.as_str()) {
                return Err("market_instrument_id must be unique".into());
            }
            if !targets.insert(instrument.target_instrument_id.as_str()) {
                return Err("target_instrument_id must be unique".into());
            }
        }
        if symbols.into_iter().collect::<Vec<_>>() != LAB0334_SYMBOLS {
            return Err("lab0334 symbols must be IC8888, IF8888, IH8888 and IM8888".into());
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

    fn instrument(symbol: &str, market: &str, target: &str) -> InstrumentConfig {
        InstrumentConfig {
            symbol: symbol.into(),
            market_instrument_id: market.into(),
            target_instrument_id: target.into(),
            parquet: PathBuf::from("/unused"),
            full_weight_contracts: 10.0,
            contract_multiplier: 300.0,
            session_bar_count: 240,
            session_end_time: "15:00:00".into(),
        }
    }

    fn valid_config() -> LiveConfig {
        LiveConfig {
            account_id: "BULLET/lab0334-sim".into(),
            bind_address: "127.0.0.1:8091".parse().unwrap(),
            history_seed_bars: 1_000_000,
            ctpd: CtpdConfig {
                base_url: "https://ctpd.example.test".into(),
                bearer_token_file: PathBuf::from("/unused"),
                stale_after_ms: 10_000,
            },
            remote_account: RemoteAccountConfig {
                allow_unauthenticated: false,
                bearer_token_file: Some(PathBuf::from("/unused")),
            },
            linkit: None,
            instruments: vec![
                instrument("IC8888", "IDX-CFFEX-IC", "IC2609"),
                instrument("IF8888", "IDX-CFFEX-IF", "IF2609"),
                instrument("IH8888", "IDX-CFFEX-IH", "IH2609"),
                instrument("IM8888", "IDX-CFFEX-IM", "IM2609"),
            ],
        }
    }

    #[test]
    fn requires_the_complete_lab0334_universe() {
        let mut config = valid_config();
        config.instruments.pop();
        assert!(config.validate().unwrap_err().contains("exactly"));
    }

    #[test]
    fn rejects_duplicate_target_instruments() {
        let mut config = valid_config();
        config.instruments[1].target_instrument_id = "IC2609".into();
        assert!(
            config
                .validate()
                .unwrap_err()
                .contains("target_instrument_id")
        );
    }
}

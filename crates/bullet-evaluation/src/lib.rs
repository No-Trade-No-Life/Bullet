//! Strategy-neutral replay of externally supplied integer exposure targets.
//!
//! Units are multiples of portfolio return exposure, NOT asset quantities or
//! contracts. This accounting surface does not replace either existing backtest.

mod accounting;
mod audit;
mod streaming;
mod validation;

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SCHEMA_VERSION: u32 = 1;
pub const NANOS_PER_DAY: u64 = 86_400_000_000_000;
pub const FIXED_SCALE: f64 = 1_000_000_000_000.0;

/// `sequence` explicitly orders availability, decision and fill at one instant.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EventTime {
    pub timestamp_ns: u64,
    pub sequence: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalAvailability {
    pub source_end: EventTime,
    pub available_at: EventTime,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetDecision {
    pub decision_time: EventTime,
    pub execution_time: EventTime,
    pub instrument: String,
    pub target_units: i64,
    pub model_vintage: Option<String>,
    pub signal_id: Option<String>,
    pub source_id: Option<String>,
    pub state_code: Option<String>,
    pub diagnostic_json: Value,
    pub causal_availability: CausalAvailability,
    pub provenance_hash: String,
}

/// An observable execution/valuation price, never a complete future OHLC bar.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MarketPoint {
    pub time: EventTime,
    pub instrument: String,
    pub price: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Accounting {
    NormalizedExposureV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalPolicy {
    KeepOpen,
    Liquidate,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationConfig {
    pub accounting: Accounting,
    pub instrument: String,
    pub one_way_cost_bps: f64,
    /// An additional cost in return space; does not change observed prices.
    pub slippage_bps: f64,
    pub terminal_policy: TerminalPolicy,
    pub sharpe_periods_per_year: f64,
    pub annualization_days_per_year: f64,
    /// Consecutive UTC midnights in epoch nanoseconds, including quiet days.
    pub evaluation_days: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationInput {
    pub schema_version: u32,
    pub config: EvaluationConfig,
    pub market: Vec<MarketPoint>,
    pub decisions: Vec<TargetDecision>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionLedgerRow {
    pub decision: TargetDecision,
    pub realized_units: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionReason {
    Rebalance,
    TerminalLiquidation,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExecutionLedgerRow {
    pub execution_time: EventTime,
    pub instrument: String,
    pub decision_index: Option<usize>,
    pub reason: ExecutionReason,
    pub from_units: i64,
    pub realized_units: i64,
    pub delta_units: i64,
    pub price: f64,
    /// Return-space cost at 1e12 scale; NOT money paid for asset quantities.
    pub cost_units: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PositionLedgerRow {
    pub start_time: EventTime,
    pub end_time: EventTime,
    pub instrument: String,
    pub realized_units: i64,
    pub turnover_units: u64,
    pub gross_return_units: i64,
    pub cost_units: i64,
    pub net_return_units: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DailyReturnRow {
    pub day_timestamp_ns: u64,
    pub gross_return_units: i64,
    pub net_return_units: i64,
    pub cost_sum_units: i64,
    pub turnover_units: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvaluationMetrics {
    pub days: usize,
    pub total_return_units: i64,
    pub mean_daily_return_units: i64,
    pub daily_volatility_units: i64,
    pub sharpe_units: i64,
    pub annualized_return_units: i64,
    /// Nonnegative magnitude, measured on compounded daily net returns.
    pub max_drawdown_units: i64,
    pub turnover_units: u64,
    pub total_cost_units: i64,
    pub completed_directional_trades: usize,
    pub ending_target_units: i64,
    pub ending_realized_units: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AuditManifest {
    pub schema_version: u32,
    pub market_point_count: usize,
    pub decision_count: usize,
    pub execution_count: usize,
    pub interval_count: usize,
    pub config_sha256: String,
    pub market_sha256: String,
    pub decision_sha256: String,
    pub ledger_sha256: String,
    pub run_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvaluationResult {
    pub schema_version: u32,
    pub config: EvaluationConfig,
    pub decision_ledger: Vec<DecisionLedgerRow>,
    pub execution_ledger: Vec<ExecutionLedgerRow>,
    pub position_ledger: Vec<PositionLedgerRow>,
    pub daily_returns: Vec<DailyReturnRow>,
    pub metrics: EvaluationMetrics,
    pub audit: AuditManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluationError(String);

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl Error for EvaluationError {}

type Result<T> = std::result::Result<T, EvaluationError>;

fn require(valid: bool, message: impl Into<String>) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(EvaluationError(message.into()))
    }
}

/// Validates declared causality; it cannot certify a producer's feature/model logic.
/// Starts flat. Decisions must match an execution point before the terminal mark.
/// No sorting, missing-price interpolation, or target clamping is performed.
pub fn evaluate(input: &EvaluationInput) -> Result<EvaluationResult> {
    validation::validate(input)?;
    accounting::replay(input)
}

pub use streaming::{
    JsonlReplayOptions, StreamOutputManifest, StreamRunSummary, StreamStatus, evaluate_jsonl,
    hash_decision_jsonl, hash_market_jsonl,
};

/// Decimal half-even formatting of binary64 to twelve places, without -0.
/// Canonical integers are presentation/audit values, not feedback into accounting.
pub fn to_fixed(value: f64) -> Result<i64> {
    require(value.is_finite(), "non-finite canonical value")?;
    let text = format!("{value:.12}");
    text.replace('.', "")
        .parse::<i64>()
        .map_err(|_| EvaluationError("canonical value exceeds signed 1e12 range".into()))
}

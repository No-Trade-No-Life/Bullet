//! Pure-Rust causal strategy runtime for Bullet normalized-exposure evaluation.
//!
//! The runtime owns the protocol plumbing that every strategy otherwise has to
//! repeat: causal history windows, decision/execution timestamps, metadata,
//! provenance hashes, calendar derivation, and the handoff to
//! `bullet-evaluation`. Strategy-specific features, models, and state machines
//! remain in the caller's `Strategy` implementation.

use std::error::Error;
use std::fmt;

use bullet_evaluation::{
    CausalAvailability, EvaluationConfig, EvaluationError, EvaluationInput, EvaluationResult,
    EventTime, MarketPoint, TargetDecision, evaluate,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const STRATEGY_RUNTIME_SCHEMA_VERSION: u32 = 1;

/// Metadata applied to decisions when an intent does not override a field.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StrategyMetadata {
    pub model_vintage: Option<String>,
    pub signal_id: Option<String>,
    pub source_id: Option<String>,
}

/// Runtime configuration. An empty evaluation calendar is derived from the
/// supplied market points; an explicit calendar is preserved and validated by
/// `bullet-evaluation`.
#[derive(Clone, Debug)]
pub struct StrategyConfig {
    pub evaluation: EvaluationConfig,
    pub metadata: StrategyMetadata,
}

/// The only market view exposed to a strategy callback.
///
/// `history` ends at `observation`; the price at `execution_time` is never
/// exposed. This makes the next-open boundary explicit instead of relying on a
/// strategy author's discipline.
#[derive(Clone, Debug)]
pub struct DecisionContext<'a> {
    pub step: usize,
    pub instrument: &'a str,
    pub history: &'a [MarketPoint],
    pub observation: &'a MarketPoint,
    pub execution_time: EventTime,
    pub previous_target_units: i64,
}

impl DecisionContext<'_> {
    /// Returns an intent for a new normalized-exposure target.
    pub fn target(&self, target_units: i64) -> TargetIntent {
        TargetIntent::new(target_units)
    }

    /// Returns an intent that carries the previous target forward.
    pub fn hold(&self) -> TargetIntent {
        self.target(self.previous_target_units)
    }
}

/// Strategy output for one execution point.
#[derive(Clone, Debug)]
pub struct TargetIntent {
    pub target_units: i64,
    pub model_vintage: Option<String>,
    pub signal_id: Option<String>,
    pub source_id: Option<String>,
    pub state_code: Option<String>,
    pub diagnostic_json: Value,
    pub causal_availability: Option<CausalAvailability>,
}

impl TargetIntent {
    pub fn new(target_units: i64) -> Self {
        Self {
            target_units,
            model_vintage: None,
            signal_id: None,
            source_id: None,
            state_code: None,
            diagnostic_json: Value::Object(serde_json::Map::new()),
            causal_availability: None,
        }
    }

    pub fn model_vintage(mut self, value: impl Into<String>) -> Self {
        self.model_vintage = Some(value.into());
        self
    }

    pub fn signal_id(mut self, value: impl Into<String>) -> Self {
        self.signal_id = Some(value.into());
        self
    }

    pub fn source_id(mut self, value: impl Into<String>) -> Self {
        self.source_id = Some(value.into());
        self
    }

    pub fn state_code(mut self, value: impl Into<String>) -> Self {
        self.state_code = Some(value.into());
        self
    }

    pub fn diagnostic(mut self, value: Value) -> Self {
        self.diagnostic_json = value;
        self
    }

    /// Overrides the default availability, which is the observation point.
    pub fn causal_availability(mut self, source_end: EventTime, available_at: EventTime) -> Self {
        self.causal_availability = Some(CausalAvailability {
            source_end,
            available_at,
        });
        self
    }
}

/// A strategy receives a causal context and returns one target intent.
pub trait Strategy {
    type Error: Error + Send + Sync + 'static;

    fn decide(&mut self, context: &DecisionContext<'_>) -> Result<TargetIntent, Self::Error>;
}

#[derive(Debug)]
pub enum StrategyRuntimeError<E> {
    InvalidMarket(String),
    Decision { execution_index: usize, source: E },
    Provenance(String),
    Evaluation(EvaluationError),
}

impl<E: fmt::Display> fmt::Display for StrategyRuntimeError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMarket(message) => formatter.write_str(message),
            Self::Decision {
                execution_index,
                source,
            } => write!(
                formatter,
                "strategy decision at execution index {execution_index}: {source}"
            ),
            Self::Provenance(message) => formatter.write_str(message),
            Self::Evaluation(error) => error.fmt(formatter),
        }
    }
}

impl<E: Error + 'static> Error for StrategyRuntimeError<E> {}

/// Pure-Rust strategy runner. It first compiles strategy callbacks into the
/// same `EvaluationInput` consumed by the low-level evaluator, then optionally
/// evaluates that input immediately.
pub struct StrategyRunner<S> {
    strategy: S,
    market: Vec<MarketPoint>,
    config: StrategyConfig,
}

impl<S: Strategy> StrategyRunner<S> {
    pub fn new(
        strategy: S,
        market: Vec<MarketPoint>,
        mut config: StrategyConfig,
    ) -> Result<Self, StrategyRuntimeError<S::Error>> {
        validate_market(&market, &config.evaluation.instrument)?;
        if config.evaluation.evaluation_days.is_empty() {
            config.evaluation.evaluation_days =
                derive_evaluation_days(&market).map_err(StrategyRuntimeError::InvalidMarket)?;
        }
        Ok(Self {
            strategy,
            market,
            config,
        })
    }

    /// Consumes the runner after compiling the strategy path exactly once.
    pub fn compile(mut self) -> Result<EvaluationInput, StrategyRuntimeError<S::Error>> {
        let mut decisions = Vec::with_capacity(self.market.len().saturating_sub(1));
        let mut previous_target_units = 0_i64;
        for execution_index in 1..self.market.len().saturating_sub(1) {
            let observation_index = execution_index - 1;
            let observation = &self.market[observation_index];
            let execution = &self.market[execution_index];
            let context = DecisionContext {
                step: observation_index,
                instrument: &self.config.evaluation.instrument,
                history: &self.market[..execution_index],
                observation,
                execution_time: execution.time,
                previous_target_units,
            };
            let intent = self.strategy.decide(&context).map_err(|source| {
                StrategyRuntimeError::Decision {
                    execution_index,
                    source,
                }
            })?;
            let decision = self.build_decision(&context, intent)?;
            previous_target_units = decision.target_units;
            decisions.push(decision);
        }
        Ok(EvaluationInput {
            schema_version: bullet_evaluation::SCHEMA_VERSION,
            config: self.config.evaluation.clone(),
            market: self.market.clone(),
            decisions,
        })
    }

    pub fn run(self) -> Result<EvaluationResult, StrategyRuntimeError<S::Error>> {
        let input = self.compile()?;
        evaluate(&input).map_err(StrategyRuntimeError::Evaluation)
    }

    pub fn evaluation_config(&self) -> &EvaluationConfig {
        &self.config.evaluation
    }

    fn build_decision(
        &self,
        context: &DecisionContext<'_>,
        mut intent: TargetIntent,
    ) -> Result<TargetDecision, StrategyRuntimeError<S::Error>> {
        intent.diagnostic_json.sort_all_objects();
        let availability = intent.causal_availability.unwrap_or(CausalAvailability {
            source_end: context.observation.time,
            available_at: context.observation.time,
        });
        let model_vintage = intent
            .model_vintage
            .or_else(|| self.config.metadata.model_vintage.clone());
        let signal_id = intent
            .signal_id
            .or_else(|| self.config.metadata.signal_id.clone());
        let source_id = intent
            .source_id
            .or_else(|| self.config.metadata.source_id.clone());
        let provenance_hash = provenance_hash(&ProvenancePayload {
            runtime_schema_version: STRATEGY_RUNTIME_SCHEMA_VERSION,
            instrument: &self.config.evaluation.instrument,
            decision_time: context.observation.time,
            execution_time: context.execution_time,
            target_units: intent.target_units,
            model_vintage: model_vintage.as_deref(),
            signal_id: signal_id.as_deref(),
            source_id: source_id.as_deref(),
            state_code: intent.state_code.as_deref(),
            diagnostic_json: &intent.diagnostic_json,
            causal_availability: &availability,
        })
        .map_err(StrategyRuntimeError::Provenance)?;
        Ok(TargetDecision {
            decision_time: context.observation.time,
            execution_time: context.execution_time,
            instrument: self.config.evaluation.instrument.clone(),
            target_units: intent.target_units,
            model_vintage,
            signal_id,
            source_id,
            state_code: intent.state_code,
            diagnostic_json: intent.diagnostic_json,
            causal_availability: availability,
            provenance_hash,
        })
    }
}

#[derive(Serialize)]
struct ProvenancePayload<'a> {
    runtime_schema_version: u32,
    instrument: &'a str,
    decision_time: EventTime,
    execution_time: EventTime,
    target_units: i64,
    model_vintage: Option<&'a str>,
    signal_id: Option<&'a str>,
    source_id: Option<&'a str>,
    state_code: Option<&'a str>,
    diagnostic_json: &'a Value,
    causal_availability: &'a CausalAvailability,
}

fn provenance_hash(payload: &ProvenancePayload<'_>) -> Result<String, String> {
    let bytes = serde_json::to_vec(payload)
        .map_err(|error| format!("cannot serialize provenance: {error}"))?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_market<S>(
    market: &[MarketPoint],
    instrument: &str,
) -> Result<(), StrategyRuntimeError<S>>
where
    S: fmt::Display,
{
    if market.len() < 2 {
        return Err(StrategyRuntimeError::InvalidMarket(
            "strategy runtime requires at least two market points".into(),
        ));
    }
    if instrument.trim().is_empty() {
        return Err(StrategyRuntimeError::InvalidMarket(
            "strategy runtime instrument must not be empty".into(),
        ));
    }
    for (index, point) in market.iter().enumerate() {
        if point.instrument != instrument {
            return Err(StrategyRuntimeError::InvalidMarket(format!(
                "market point {index} instrument differs from strategy config"
            )));
        }
        if !point.price.is_finite() || point.price <= 0.0 {
            return Err(StrategyRuntimeError::InvalidMarket(format!(
                "market point {index} price is not finite and positive"
            )));
        }
        if index > 0 && market[index - 1].time.timestamp_ns >= point.time.timestamp_ns {
            return Err(StrategyRuntimeError::InvalidMarket(
                "market timestamps must be strictly increasing".into(),
            ));
        }
    }
    Ok(())
}

fn derive_evaluation_days(market: &[MarketPoint]) -> Result<Vec<u64>, String> {
    let first = market
        .first()
        .ok_or_else(|| "market is empty".to_owned())?
        .time
        .timestamp_ns
        / bullet_evaluation::NANOS_PER_DAY
        * bullet_evaluation::NANOS_PER_DAY;
    let last = market
        .last()
        .ok_or_else(|| "market is empty".to_owned())?
        .time
        .timestamp_ns
        / bullet_evaluation::NANOS_PER_DAY
        * bullet_evaluation::NANOS_PER_DAY;
    let mut days = Vec::new();
    let mut day = first;
    loop {
        days.push(day);
        if day == last {
            break;
        }
        day = day
            .checked_add(bullet_evaluation::NANOS_PER_DAY)
            .ok_or_else(|| "evaluation calendar overflows".to_owned())?;
        if day > last {
            return Err("evaluation calendar is not consecutive".into());
        }
    }
    Ok(days)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::convert::Infallible;

    struct Momentum;

    impl Strategy for Momentum {
        type Error = Infallible;

        fn decide(&mut self, context: &DecisionContext<'_>) -> Result<TargetIntent, Self::Error> {
            let previous = context.history[context.history.len() - 1].price;
            Ok(context
                .target(if previous > 100.0 { 1 } else { 0 })
                .state_code("momentum")
                .diagnostic(json!({"observed_price": previous})))
        }
    }

    fn config() -> StrategyConfig {
        StrategyConfig {
            evaluation: EvaluationConfig {
                accounting: bullet_evaluation::Accounting::NormalizedExposureV1,
                instrument: "TEST".into(),
                one_way_cost_bps: 0.0,
                slippage_bps: 0.0,
                terminal_policy: bullet_evaluation::TerminalPolicy::Liquidate,
                sharpe_periods_per_year: 1.0,
                sharpe_standard_deviation_ddof: 1,
                annualization_days_per_year: 1.0,
                evaluation_days: Vec::new(),
            },
            metadata: StrategyMetadata {
                model_vintage: Some("momentum-v1".into()),
                signal_id: Some("test-signal".into()),
                source_id: Some("test-source".into()),
            },
        }
    }

    fn market() -> Vec<MarketPoint> {
        [100.0, 101.0, 99.0, 102.0]
            .into_iter()
            .enumerate()
            .map(|(index, price)| MarketPoint {
                time: EventTime {
                    timestamp_ns: (index as u64 + 1) * 1_000,
                    sequence: 0,
                },
                instrument: "TEST".into(),
                price,
            })
            .collect()
    }

    #[test]
    fn compiles_a_strategy_into_causal_decisions_and_evaluates() {
        let runner = StrategyRunner::new(Momentum, market(), config()).expect("runner");
        let input = runner.compile().expect("strategy input");

        assert_eq!(input.decisions.len(), 2);
        assert_eq!(input.config.evaluation_days, vec![0]);
        assert_eq!(input.decisions[0].target_units, 0);
        assert_eq!(input.decisions[1].target_units, 1);
        assert_eq!(
            input.decisions[1].causal_availability.available_at,
            input.decisions[1].decision_time
        );
        assert!(input.decisions[1].decision_time < input.decisions[1].execution_time);
        assert_eq!(
            input.decisions[1].model_vintage.as_deref(),
            Some("momentum-v1")
        );
        assert_eq!(input.decisions[1].state_code.as_deref(), Some("momentum"));

        let result = StrategyRunner::new(Momentum, market(), config())
            .expect("runner")
            .run()
            .expect("evaluation");
        assert_eq!(result.metrics.completed_directional_trades, 1);
        assert_eq!(result.metrics.ending_realized_units, 0);
    }

    #[test]
    fn history_excludes_the_execution_price() {
        struct HistoryLength;

        impl Strategy for HistoryLength {
            type Error = Infallible;

            fn decide(
                &mut self,
                context: &DecisionContext<'_>,
            ) -> Result<TargetIntent, Self::Error> {
                assert_eq!(context.history.len(), context.step + 1);
                assert_eq!(
                    context.history.last().unwrap().time,
                    context.observation.time
                );
                Ok(context.target(0))
            }
        }

        let runner = StrategyRunner::new(HistoryLength, market(), config()).expect("runner");
        runner.compile().expect("strategy input");
    }

    #[test]
    fn explicit_calendar_is_preserved() {
        let mut value = config();
        value.evaluation.evaluation_days = vec![0, bullet_evaluation::NANOS_PER_DAY];
        let runner = StrategyRunner::new(Momentum, market(), value).expect("runner");
        assert_eq!(
            runner.config.evaluation.evaluation_days,
            vec![0, bullet_evaluation::NANOS_PER_DAY]
        );
    }
}

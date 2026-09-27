use std::convert::Infallible;

use bullet_evaluation::{Accounting, EvaluationConfig, EventTime, MarketPoint, TerminalPolicy};
use bullet_strategy::{
    DecisionContext, Strategy, StrategyConfig, StrategyMetadata, StrategyRunner, TargetIntent,
};

struct Momentum;

impl Strategy for Momentum {
    type Error = Infallible;

    fn decide(&mut self, context: &DecisionContext<'_>) -> Result<TargetIntent, Self::Error> {
        let observed_price = context.observation.price;
        Ok(context
            .target(i64::from(observed_price > 100.0))
            .model_vintage("momentum-v1")
            .diagnostic(serde_json::json!({"observed_price": observed_price})))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let market = [100.0, 101.0, 99.0, 102.0]
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
        .collect();
    let config = StrategyConfig {
        evaluation: EvaluationConfig {
            accounting: Accounting::NormalizedExposureV1,
            instrument: "TEST".into(),
            one_way_cost_bps: 4.0,
            slippage_bps: 0.0,
            terminal_policy: TerminalPolicy::Liquidate,
            sharpe_periods_per_year: 365.0,
            sharpe_standard_deviation_ddof: 1,
            annualization_days_per_year: 365.2425,
            evaluation_days: Vec::new(),
        },
        metadata: StrategyMetadata::default(),
    };
    let result = StrategyRunner::new(Momentum, market, config)?.run()?;
    println!("sharpe_units={}", result.metrics.sharpe_units);
    Ok(())
}

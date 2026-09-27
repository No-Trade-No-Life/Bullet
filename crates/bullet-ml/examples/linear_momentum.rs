use bullet_evaluation::{Accounting, EvaluationConfig, EventTime, MarketPoint, TerminalPolicy};
use bullet_ml::{
    FeaturePipeline, LinearModel, MlStrategy, ModelMetadata, RollingPriceFeatures, ScoreToExposure,
    WarmupPolicy,
};
use bullet_strategy::{StrategyConfig, StrategyMetadata, StrategyRunner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let market = [100.0, 101.0, 102.0, 99.0, 103.0, 104.0]
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
    let pipeline = RollingPriceFeatures::new(vec![1, 2])?;
    let model = LinearModel::new(
        ModelMetadata::new("linear-momentum-v1", "linear"),
        pipeline.schema().clone(),
        vec![100.0, 25.0],
        0.0,
    )?;
    let mapper = ScoreToExposure::new(0.01, -0.01, 1, -1)?;
    let strategy = MlStrategy::try_new(pipeline, model, mapper, WarmupPolicy::Hold)?;
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
    let result = StrategyRunner::new(strategy, market, config)?.run()?;
    println!("sharpe_units={}", result.metrics.sharpe_units);
    Ok(())
}

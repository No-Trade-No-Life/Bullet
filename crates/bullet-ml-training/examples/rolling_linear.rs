//! Bounded weighted LogisticRegression/Huber -> scheduled OOS ledger.
//! This example is not an asset strategy, a profitability claim, or SOTA parity evidence.
use bullet_evaluation::{
    Accounting, EvaluationConfig, EventTime, MarketPoint, TerminalPolicy, evaluate,
};
use bullet_ml::{
    FeatureError, FeatureOutput, FeaturePipeline, FeatureSchema, FeatureVector, PreprocessedModel,
    ScheduledMlStrategy, ScoreToExposure, WarmupPolicy,
};
use bullet_ml_training::{
    TrainingDataset, TrainingError, TrainingExample,
    linear_estimators::{HuberRegressionTrainer, LogisticRegressionTrainer, SOLVER_VERSION},
    rolling::{
        ArtifactModel, FitBatch, FitReceipt, FittedFold, FoldReport, RollingOptions, RollingPlan,
        RollingTrainer, Scaling, train_rolling,
    },
};
use bullet_strategy::{DecisionContext, StrategyConfig, StrategyMetadata, StrategyRunner};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;
use std::{collections::VecDeque, error::Error, path::PathBuf, time::Instant};

const TICK_NS: u64 = 86_400_000_000_000;
struct Returns {
    schema: FeatureSchema,
    prices: VecDeque<f64>,
}
fn values(prices: &[f64]) -> Vec<f64> {
    vec![prices[3] / prices[2] - 1.0, prices[3] / prices[0] - 1.0]
}
impl FeaturePipeline for Returns {
    type Error = FeatureError;
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }
    fn extract(&mut self, context: &DecisionContext<'_>) -> Result<FeatureOutput, Self::Error> {
        self.prices.push_back(context.observation.price);
        while self.prices.len() > 4 {
            self.prices.pop_front();
        }
        if self.prices.len() < 4 {
            return Ok(FeatureOutput::Warmup {
                required_history: 4,
            });
        }
        let p: Vec<_> = self.prices.iter().copied().collect();
        Ok(FeatureOutput::Ready(FeatureVector::new(
            self.schema.clone(),
            values(&p),
        )?))
    }
}
fn config() -> StrategyConfig {
    StrategyConfig {
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
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: rolling_linear <logistic|huber> <rows:64..4096, divisible by 8> <new-output-dir>".into());
    }
    let n: usize = args[2].parse()?;
    if !(64..=4096).contains(&n) || !n.is_multiple_of(8) {
        return Err("invalid bounded row count".into());
    }
    let output = PathBuf::from(&args[3]);
    match args[1].as_str() {
        "logistic" => run(
            n,
            output,
            "logistic",
            LogisticRegressionTrainer::new(Default::default())?,
            |current, next| f64::from(next > current),
            ScoreToExposure::new(0.55, 0.45, 1, -1)?,
        ),
        "huber" => run(
            n,
            output,
            "huber",
            HuberRegressionTrainer::new(Default::default())?,
            |current, next| next - current,
            ScoreToExposure::new(0.01, -0.01, 1, -1)?,
        ),
        _ => Err("unknown linear estimator".into()),
    }
}
struct ProgressTrainer<B> {
    backend: B,
    completed: usize,
    total: usize,
    started: Instant,
}
impl<B: RollingTrainer> RollingTrainer for ProgressTrainer<B> {
    type Model = B::Model;
    fn fit(&mut self, batch: &FitBatch<'_>) -> Result<(Self::Model, FitReceipt), TrainingError> {
        let (model, receipt) = self.backend.fit(batch)?;
        self.completed += 1;
        eprintln!(
            "PERF_PROGRESS processed={} total={} elapsed_sec={:.9} unit=folds rows={} features={} iterations={} evaluations={}",
            self.completed,
            self.total,
            self.started.elapsed().as_secs_f64(),
            batch.features.len(),
            batch.schema.names.len(),
            receipt.parameters["optimizer"]["iterations"],
            receipt.parameters["optimizer"]["evaluations"]
        );
        Ok((model, receipt))
    }
}
fn run<B: RollingTrainer>(
    n: usize,
    output: PathBuf,
    family: &str,
    backend: B,
    label: impl Fn(f64, f64) -> f64,
    mapper: ScoreToExposure,
) -> Result<(), Box<dyn Error>>
where
    B::Model: Serialize + DeserializeOwned,
{
    std::fs::create_dir(&output)?;
    let started = Instant::now();
    let market: Vec<_> = (0..n + 2)
        .map(|i| MarketPoint {
            time: EventTime {
                timestamp_ns: i as u64 * TICK_NS,
                sequence: 0,
            },
            instrument: "TEST".into(),
            price: 100.0 + 2.0 * (i as f64 * 0.21).sin() + 0.7 * (i as f64 * 0.073).cos(),
        })
        .collect();
    let schema = FeatureSchema::new(vec!["return_1".into(), "return_3".into()])?;
    let rows = (3..n)
        .map(|i| {
            let prices: Vec<_> = market[i - 3..=i].iter().map(|p| p.price).collect();
            Ok(TrainingExample::new(
                market[i].time.timestamp_ns,
                market[i].time.timestamp_ns,
                market[i + 1].time.timestamp_ns,
                FeatureVector::new(schema.clone(), values(&prices))?,
                label(market[i].price, market[i + 1].price),
            )?)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let dataset = TrainingDataset::new(schema.clone(), rows)?;
    let weights: Vec<_> = (0..dataset.len())
        .map(|i| if i % 3 == 0 { 2.0 } else { 1.0 })
        .collect();
    let first = n / 2;
    let plan = RollingPlan::new(
        first as u64 * TICK_NS,
        n as u64 * TICK_NS,
        (n / 4) as u64 * TICK_NS,
        (n / 8) as u64 * TICK_NS,
        0,
        0,
    )?;
    let options = RollingOptions {
        minimum_training_samples: 8,
        scaling: Scaling::Standardize,
        weight_normalization: bullet_ml_training::rolling::WeightNormalization::TrainingMeanOne,
        vintage_prefix: format!("synthetic-{family}"),
    };
    let mut backend = ProgressTrainer {
        backend,
        completed: 0,
        total: plan.folds().len(),
        started,
    };
    let folds = train_rolling(&dataset, &weights, &plan, &options, &mut backend)?;
    let training_sec = started.elapsed().as_secs_f64();
    let workloads: Vec<_> = folds
        .iter()
        .map(|fold| {
            json!({
                "training_rows": fold.report.training_samples,
                "features": schema.names.len(),
                "optimizer": fold.report.receipt.parameters["optimizer"],
            })
        })
        .collect();
    let mut restored = Vec::new();
    for (index, fold) in folds.iter().enumerate() {
        let artifact_path = output.join(format!("model-{index}.json"));
        let report_path = output.join(format!("fold-{index}.json"));
        std::fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(fold.model.model())?,
        )?;
        std::fs::write(&report_path, serde_json::to_vec_pretty(&fold.report)?)?;
        let report: FoldReport = serde_json::from_slice(&std::fs::read(report_path)?)?;
        report.validate()?;
        let artifact: B::Model = serde_json::from_slice(&std::fs::read(artifact_path)?)?;
        if artifact.artifact_sha256()? != report.receipt.model_sha256 {
            return Err("artifact/report hash mismatch".into());
        }
        restored.push(FittedFold {
            model: PreprocessedModel::new(artifact, report.transform.clone())?,
            report,
        });
    }
    let compile = |folds: Vec<FittedFold<B::Model>>| -> Result<_, Box<dyn Error>> {
        let windows = folds
            .into_iter()
            .map(|f| f.into_window())
            .collect::<Result<Vec<_>, _>>()?;
        let pipeline = Returns {
            schema: schema.clone(),
            prices: market[first - 3..first].iter().map(|p| p.price).collect(),
        };
        let strategy =
            ScheduledMlStrategy::new(pipeline, windows, mapper.clone(), WarmupPolicy::Hold)?;
        Ok(StrategyRunner::new(strategy, market[first..].to_vec(), config())?.compile()?)
    };
    let original = compile(folds)?;
    let reloaded = compile(restored)?;
    assert_eq!(
        serde_json::to_vec(&original)?,
        serde_json::to_vec(&reloaded)?
    );
    assert_eq!(original.decisions.len(), n - first);
    for decision in &original.decisions {
        let ml = &decision.diagnostic_json["ml"];
        assert!(decision.decision_time.timestamp_ns >= ml["valid_from_ns"].as_u64().unwrap());
        assert!(decision.decision_time.timestamp_ns < ml["valid_until_ns"].as_u64().unwrap());
        assert!(
            ml["training_report_sha256"]
                .as_str()
                .is_some_and(|s| s.len() == 64)
        );
    }
    let result = evaluate(&original)?;
    let restored_result = evaluate(&reloaded)?;
    assert_eq!(result, restored_result);
    std::fs::write(
        output.join("ledger.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    let summary = json!({"scope":"bounded synthetic linear rolling integration, not SOTA training parity","family":family,
        "market_rows":n,"training_folds":plan.folds().len(),"features":2,
        "fold_workloads":workloads,"training_sec":training_sec,"elapsed_sec":started.elapsed().as_secs_f64(),
        "oos_decisions":original.decisions.len(),"artifact_reload_decisions_and_ledger_exact":true,
        "audit":result.audit,"solver":SOLVER_VERSION});
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{}", summary);
    Ok(())
}

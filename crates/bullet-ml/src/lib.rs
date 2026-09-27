//! Pure-Rust ML building blocks for Bullet causal strategies.
//!
//! This crate does not train models. It standardizes the inference boundary:
//! causal feature extraction, feature-schema validation, model prediction,
//! prediction-to-target mapping, warmup handling, and ML provenance metadata.
//! The resulting strategy still runs through `bullet-strategy` and
//! `bullet-evaluation`.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use bullet_strategy::{DecisionContext, Strategy, TargetIntent};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FeatureSchema {
    pub names: Vec<String>,
    pub sha256: String,
}

impl FeatureSchema {
    pub fn new(names: Vec<String>) -> Result<Self, FeatureError> {
        if names.is_empty() {
            return Err(FeatureError("feature schema is empty".into()));
        }
        if names.iter().any(|name| name.trim().is_empty()) {
            return Err(FeatureError("feature names must not be empty".into()));
        }
        for (index, name) in names.iter().enumerate() {
            if names[..index].iter().any(|previous| previous == name) {
                return Err(FeatureError(format!("duplicate feature name: {name}")));
            }
        }
        let sha256 = feature_schema_hash(&names)?;
        Ok(Self { names, sha256 })
    }

    pub fn validate(&self) -> Result<(), FeatureError> {
        let expected = Self::new(self.names.clone())?;
        if expected.sha256 != self.sha256 {
            return Err(FeatureError(
                "feature schema hash does not match feature names".into(),
            ));
        }
        Ok(())
    }
}

fn feature_schema_hash(names: &[String]) -> Result<String, FeatureError> {
    let bytes = serde_json::to_vec(names)
        .map_err(|error| FeatureError(format!("cannot serialize feature schema: {error}")))?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Clone, Debug, Serialize)]
pub struct FeatureVector {
    pub schema: FeatureSchema,
    pub values: Vec<f64>,
}

impl FeatureVector {
    pub fn new(schema: FeatureSchema, values: Vec<f64>) -> Result<Self, FeatureError> {
        let vector = Self { schema, values };
        vector.validate()?;
        Ok(vector)
    }

    pub fn validate(&self) -> Result<(), FeatureError> {
        self.schema.validate()?;
        if self.schema.names.len() != self.values.len() {
            return Err(FeatureError(format!(
                "feature dimension mismatch: schema={} values={}",
                self.schema.names.len(),
                self.values.len()
            )));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(FeatureError("feature values must be finite".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum FeatureOutput {
    Warmup { required_history: usize },
    Ready(FeatureVector),
}

pub trait FeaturePipeline {
    type Error: Error + Send + Sync + 'static;

    fn schema(&self) -> &FeatureSchema;
    fn extract(&mut self, context: &DecisionContext<'_>) -> Result<FeatureOutput, Self::Error>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_vintage: String,
    pub model_type: String,
}

impl ModelMetadata {
    pub fn new(model_vintage: impl Into<String>, model_type: impl Into<String>) -> Self {
        Self {
            model_vintage: model_vintage.into(),
            model_type: model_type.into(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Prediction {
    pub score: f64,
    pub diagnostic_json: Value,
}

impl Prediction {
    pub fn score(score: f64) -> Result<Self, ModelError> {
        if !score.is_finite() {
            return Err(ModelError("prediction score must be finite".into()));
        }
        Ok(Self {
            score,
            diagnostic_json: Value::Object(serde_json::Map::new()),
        })
    }

    pub fn diagnostic(mut self, value: Value) -> Self {
        self.diagnostic_json = value;
        self
    }
}

pub trait Model {
    type Error: Error + Send + Sync + 'static;

    fn metadata(&self) -> &ModelMetadata;
    fn feature_schema(&self) -> &FeatureSchema;
    fn predict(&mut self, features: &FeatureVector) -> Result<Prediction, Self::Error>;
}

pub trait TargetMapper {
    type Error: Error + Send + Sync + 'static;

    fn map(
        &mut self,
        context: &DecisionContext<'_>,
        prediction: &Prediction,
    ) -> Result<TargetIntent, Self::Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WarmupPolicy {
    Hold,
    Flat,
}

#[derive(Debug)]
pub enum MlStrategyConfigError {
    InvalidFeatureSchema {
        component: &'static str,
        message: String,
    },
    FeatureSchemaMismatch {
        pipeline: String,
        model: String,
    },
}

impl fmt::Display for MlStrategyConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFeatureSchema { component, message } => {
                write!(formatter, "invalid {component} feature schema: {message}")
            }
            Self::FeatureSchemaMismatch { pipeline, model } => write!(
                formatter,
                "feature schema mismatch: pipeline={pipeline} model={model}"
            ),
        }
    }
}

impl Error for MlStrategyConfigError {}

#[derive(Debug)]
pub enum MlStrategyError<FE, ME, TE> {
    Features(FE),
    Model(ME),
    Mapping(TE),
}

impl<FE: fmt::Display, ME: fmt::Display, TE: fmt::Display> fmt::Display
    for MlStrategyError<FE, ME, TE>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Features(error) => write!(formatter, "feature pipeline: {error}"),
            Self::Model(error) => write!(formatter, "model inference: {error}"),
            Self::Mapping(error) => write!(formatter, "target mapping: {error}"),
        }
    }
}

impl<FE: Error + 'static, ME: Error + 'static, TE: Error + 'static> Error
    for MlStrategyError<FE, ME, TE>
{
}

#[derive(Debug)]
pub struct MlStrategy<P, M, T> {
    pipeline: P,
    model: M,
    mapper: T,
    warmup_policy: WarmupPolicy,
}

impl<P, M, T> MlStrategy<P, M, T>
where
    P: FeaturePipeline,
    M: Model,
    T: TargetMapper,
{
    pub fn try_new(
        pipeline: P,
        model: M,
        mapper: T,
        warmup_policy: WarmupPolicy,
    ) -> Result<Self, MlStrategyConfigError> {
        pipeline.schema().validate().map_err(|error| {
            MlStrategyConfigError::InvalidFeatureSchema {
                component: "pipeline",
                message: error.to_string(),
            }
        })?;
        model.feature_schema().validate().map_err(|error| {
            MlStrategyConfigError::InvalidFeatureSchema {
                component: "model",
                message: error.to_string(),
            }
        })?;
        if pipeline.schema().sha256 != model.feature_schema().sha256 {
            return Err(MlStrategyConfigError::FeatureSchemaMismatch {
                pipeline: pipeline.schema().sha256.clone(),
                model: model.feature_schema().sha256.clone(),
            });
        }
        Ok(Self {
            pipeline,
            model,
            mapper,
            warmup_policy,
        })
    }
}

impl<P, M, T> Strategy for MlStrategy<P, M, T>
where
    P: FeaturePipeline,
    M: Model,
    T: TargetMapper,
{
    type Error = MlStrategyError<P::Error, M::Error, T::Error>;

    fn decide(&mut self, context: &DecisionContext<'_>) -> Result<TargetIntent, Self::Error> {
        let output = self
            .pipeline
            .extract(context)
            .map_err(MlStrategyError::Features)?;
        let model_metadata = self.model.metadata().clone();
        let (mut intent, ml_diagnostic) = match output {
            FeatureOutput::Warmup { required_history } => {
                let target = match self.warmup_policy {
                    WarmupPolicy::Hold => context.hold(),
                    WarmupPolicy::Flat => context.target(0),
                };
                (
                    target.state_code("ml_warmup"),
                    json!({
                        "status": "warmup",
                        "required_history": required_history,
                        "model_vintage": model_metadata.model_vintage,
                        "model_type": model_metadata.model_type,
                    }),
                )
            }
            FeatureOutput::Ready(features) => {
                let prediction = self
                    .model
                    .predict(&features)
                    .map_err(MlStrategyError::Model)?;
                let intent = self
                    .mapper
                    .map(context, &prediction)
                    .map_err(MlStrategyError::Mapping)?;
                let mut diagnostic = json!({
                    "status": "ready",
                    "feature_schema_sha256": features.schema.sha256,
                    "feature_count": features.values.len(),
                    "model_vintage": model_metadata.model_vintage,
                    "model_type": model_metadata.model_type,
                    "prediction": prediction,
                });
                if let Value::Object(ref mut object) = diagnostic {
                    object.insert("mapper_diagnostic".into(), intent.diagnostic_json.clone());
                }
                (intent, diagnostic)
            }
        };
        if intent.model_vintage.is_none() {
            intent.model_vintage = Some(model_metadata.model_vintage);
        }
        intent.diagnostic_json = merge_diagnostics(intent.diagnostic_json, ml_diagnostic);
        Ok(intent)
    }
}

fn merge_diagnostics(original: Value, ml: Value) -> Value {
    match original {
        Value::Object(mut object) => {
            object.insert("ml".into(), ml);
            Value::Object(object)
        }
        other => json!({"strategy": other, "ml": ml}),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeatureError(pub String);

impl fmt::Display for FeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for FeatureError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ModelError {}

#[derive(Clone, Debug)]
pub struct RollingPriceFeatures {
    lookbacks: Vec<usize>,
    schema: FeatureSchema,
}

impl RollingPriceFeatures {
    pub fn new(lookbacks: Vec<usize>) -> Result<Self, FeatureError> {
        if lookbacks.is_empty() || lookbacks.contains(&0) {
            return Err(FeatureError(
                "rolling-price lookbacks must be positive and nonempty".into(),
            ));
        }
        let mut names = Vec::with_capacity(lookbacks.len());
        for lookback in &lookbacks {
            names.push(format!("open_to_open_return_{lookback}"));
        }
        let schema = FeatureSchema::new(names)?;
        Ok(Self { lookbacks, schema })
    }
}

impl FeaturePipeline for RollingPriceFeatures {
    type Error = FeatureError;

    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    fn extract(&mut self, context: &DecisionContext<'_>) -> Result<FeatureOutput, Self::Error> {
        let required_history = self.lookbacks.iter().copied().max().unwrap_or(0) + 1;
        if context.history.len() < required_history {
            return Ok(FeatureOutput::Warmup { required_history });
        }
        let current = context.observation.price;
        let values = self
            .lookbacks
            .iter()
            .map(|lookback| {
                let previous = context.history[context.history.len() - 1 - lookback].price;
                let value = current / previous - 1.0;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(FeatureError("rolling return is not finite".into()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FeatureOutput::Ready(FeatureVector::new(
            self.schema.clone(),
            values,
        )?))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearModel {
    pub metadata: ModelMetadata,
    pub feature_schema: FeatureSchema,
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearModel {
    pub fn new(
        metadata: ModelMetadata,
        feature_schema: FeatureSchema,
        weights: Vec<f64>,
        bias: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            metadata,
            feature_schema,
            weights,
            bias,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.feature_schema
            .validate()
            .map_err(|error| ModelError(error.to_string()))?;
        if self.feature_schema.names.len() != self.weights.len() {
            return Err(ModelError("linear model weight dimension mismatch".into()));
        }
        if self.weights.iter().any(|weight| !weight.is_finite()) || !self.bias.is_finite() {
            return Err(ModelError("linear model parameters must be finite".into()));
        }
        Ok(())
    }
}

impl Model for LinearModel {
    type Error = ModelError;

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn feature_schema(&self) -> &FeatureSchema {
        &self.feature_schema
    }

    fn predict(&mut self, features: &FeatureVector) -> Result<Prediction, Self::Error> {
        self.validate()?;
        features
            .validate()
            .map_err(|error| ModelError(error.to_string()))?;
        if features.schema != self.feature_schema {
            return Err(ModelError(
                "linear model received an unknown feature schema".into(),
            ));
        }
        let score = self
            .weights
            .iter()
            .zip(&features.values)
            .map(|(weight, value)| weight * value)
            .sum::<f64>()
            + self.bias;
        Prediction::score(score)
    }
}

#[derive(Clone, Debug)]
pub struct ScoreToExposure {
    pub long_threshold: f64,
    pub short_threshold: f64,
    pub long_units: i64,
    pub short_units: i64,
}

impl ScoreToExposure {
    pub fn new(
        long_threshold: f64,
        short_threshold: f64,
        long_units: i64,
        short_units: i64,
    ) -> Result<Self, MappingError> {
        if !long_threshold.is_finite()
            || !short_threshold.is_finite()
            || short_threshold >= long_threshold
        {
            return Err(MappingError(
                "thresholds must be finite and short < long".into(),
            ));
        }
        if long_units <= 0 || short_units >= 0 {
            return Err(MappingError(
                "long_units must be positive and short_units negative".into(),
            ));
        }
        Ok(Self {
            long_threshold,
            short_threshold,
            long_units,
            short_units,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappingError(pub String);

impl fmt::Display for MappingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for MappingError {}

impl TargetMapper for ScoreToExposure {
    type Error = Infallible;

    fn map(
        &mut self,
        _context: &DecisionContext<'_>,
        prediction: &Prediction,
    ) -> Result<TargetIntent, Self::Error> {
        let units = if prediction.score >= self.long_threshold {
            self.long_units
        } else if prediction.score <= self.short_threshold {
            self.short_units
        } else {
            0
        };
        Ok(TargetIntent::new(units).diagnostic(json!({
            "mapper": "score_to_exposure",
            "score": prediction.score,
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_evaluation::{Accounting, EvaluationConfig, EventTime, MarketPoint, TerminalPolicy};
    use bullet_strategy::{StrategyConfig, StrategyMetadata, StrategyRunner};

    fn config() -> StrategyConfig {
        StrategyConfig {
            evaluation: EvaluationConfig {
                accounting: Accounting::NormalizedExposureV1,
                instrument: "TEST".into(),
                one_way_cost_bps: 0.0,
                slippage_bps: 0.0,
                terminal_policy: TerminalPolicy::Liquidate,
                sharpe_periods_per_year: 1.0,
                sharpe_standard_deviation_ddof: 1,
                annualization_days_per_year: 1.0,
                evaluation_days: Vec::new(),
            },
            metadata: StrategyMetadata::default(),
        }
    }

    fn market() -> Vec<MarketPoint> {
        [100.0, 101.0, 102.0, 99.0, 103.0]
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
    fn rolling_features_are_causal_and_warm_up_explicitly() {
        let mut features = RollingPriceFeatures::new(vec![1, 2]).expect("features");
        let context = DecisionContext {
            step: 0,
            instrument: "TEST",
            history: &market()[..1],
            observation: &market()[0],
            execution_time: EventTime {
                timestamp_ns: 2_000,
                sequence: 0,
            },
            previous_target_units: 0,
        };
        assert!(matches!(
            features.extract(&context).expect("feature extraction"),
            FeatureOutput::Warmup {
                required_history: 3
            }
        ));
    }

    #[test]
    fn linear_model_and_threshold_mapper_drive_a_rust_ml_strategy() {
        let pipeline = RollingPriceFeatures::new(vec![1]).expect("features");
        let schema = pipeline.schema().clone();
        let model = LinearModel::new(
            ModelMetadata::new("linear-v1", "linear"),
            schema,
            vec![100.0],
            0.0,
        )
        .expect("model");
        let mapper = ScoreToExposure::new(0.005, -0.005, 1, -1).expect("mapper");
        let strategy =
            MlStrategy::try_new(pipeline, model, mapper, WarmupPolicy::Hold).expect("ML strategy");
        let result = StrategyRunner::new(strategy, market(), config())
            .expect("runner")
            .run()
            .expect("evaluation");

        assert_eq!(result.metrics.ending_realized_units, 0);
        assert!(
            result
                .decision_ledger
                .iter()
                .any(|row| row.decision.state_code.as_deref() == Some("ml_warmup"))
        );
        assert!(
            result
                .decision_ledger
                .iter()
                .any(|row| row.decision.model_vintage.as_deref() == Some("linear-v1"))
        );
    }

    #[test]
    fn schema_mismatch_is_rejected_before_replay() {
        let pipeline = RollingPriceFeatures::new(vec![1]).expect("features");
        let other_schema = FeatureSchema::new(vec!["different".into()]).expect("schema");
        let model = LinearModel::new(
            ModelMetadata::new("linear-v1", "linear"),
            other_schema,
            vec![1.0],
            0.0,
        )
        .expect("model");
        let error = MlStrategy::try_new(
            pipeline,
            model,
            ScoreToExposure::new(0.5, -0.5, 1, -1).expect("mapper"),
            WarmupPolicy::Flat,
        )
        .expect_err("schema mismatch");
        assert!(error.to_string().contains("feature schema mismatch"));
    }
}

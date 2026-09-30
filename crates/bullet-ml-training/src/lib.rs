//! Rust causal ML training orchestration with optional native backends for Bullet.
//!
//! The training layer is deliberately chronological: it rejects malformed
//! feature/label time ordering, never shuffles rows, and makes the train/
//! validation boundary explicit. Backends convert a validated dataset into
//! the serializable inference models provided by `bullet-ml`.

#[cfg(feature = "linear-backend")]
mod lbfgsb_solver;
#[cfg(feature = "linear-backend")]
pub mod linear_estimators;
pub mod rolling;
#[cfg(feature = "xgboost-backend")]
pub mod xgboost;

use std::error::Error;
use std::fmt;
use std::ops::Range;

use sha2::{Digest, Sha256};

use bullet_ml::{FeatureSchema, FeatureVector, LinearModel};
#[cfg(feature = "smartcore-backend")]
use bullet_ml::{ModelError, ModelMetadata};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize)]
pub struct TrainingExample {
    pub feature_end_ns: u64,
    pub decision_time_ns: u64,
    pub label_end_ns: u64,
    pub features: FeatureVector,
    pub target: f64,
}

impl TrainingExample {
    pub fn new(
        feature_end_ns: u64,
        decision_time_ns: u64,
        label_end_ns: u64,
        features: FeatureVector,
        target: f64,
    ) -> Result<Self, TrainingError> {
        features
            .validate()
            .map_err(|e| TrainingError(e.to_string()))?;
        if feature_end_ns > decision_time_ns {
            return Err(TrainingError(
                "feature_end_ns must not be after decision_time_ns".into(),
            ));
        }
        if decision_time_ns >= label_end_ns {
            return Err(TrainingError(
                "decision_time_ns must be before label_end_ns".into(),
            ));
        }
        if !target.is_finite() {
            return Err(TrainingError("training target must be finite".into()));
        }
        Ok(Self {
            feature_end_ns,
            decision_time_ns,
            label_end_ns,
            features,
            target,
        })
    }
}

#[derive(Clone, Debug)]
pub struct TrainingDataset {
    schema: FeatureSchema,
    examples: Vec<TrainingExample>,
}

impl TrainingDataset {
    pub fn new(
        schema: FeatureSchema,
        examples: Vec<TrainingExample>,
    ) -> Result<Self, TrainingError> {
        schema
            .validate()
            .map_err(|error| TrainingError(error.to_string()))?;
        if examples.is_empty() {
            return Err(TrainingError("training dataset is empty".into()));
        }
        let mut previous_decision_time = None;
        for (index, example) in examples.iter().enumerate() {
            TrainingExample::new(
                example.feature_end_ns,
                example.decision_time_ns,
                example.label_end_ns,
                example.features.clone(),
                example.target,
            )?;
            if example.features.schema.sha256 != schema.sha256 {
                return Err(TrainingError(format!(
                    "training row {index} has a different feature schema"
                )));
            }
            if previous_decision_time.is_some_and(|time| time >= example.decision_time_ns) {
                return Err(TrainingError(
                    "training decision times must be strictly increasing".into(),
                ));
            }
            previous_decision_time = Some(example.decision_time_ns);
        }
        Ok(Self { schema, examples })
    }

    pub fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    pub fn examples(&self) -> &[TrainingExample] {
        &self.examples
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub fn sha256(&self) -> Result<String, TrainingError> {
        let bytes = serde_json::to_vec(&self.examples).map_err(|error| {
            TrainingError(format!("cannot serialize training dataset: {error}"))
        })?;
        let mut digest = Sha256::new();
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate_split(&self, split: ChronologicalSplit) -> Result<(), TrainingError> {
        if split.validation_end_exclusive > self.examples.len()
            || split.train_end_exclusive == 0
            || split.train_end_exclusive >= split.validation_end_exclusive
        {
            return Err(TrainingError(
                "chronological split is invalid for dataset".into(),
            ));
        }
        let first_validation = &self.examples[split.train_end_exclusive];
        if self.examples[..split.train_end_exclusive]
            .iter()
            .any(|row| row.label_end_ns > first_validation.decision_time_ns)
        {
            return Err(TrainingError(
                "training label crosses the validation feature boundary".into(),
            ));
        }
        Ok(())
    }

    pub fn range(&self, range: Range<usize>) -> Result<TrainingView<'_>, TrainingError> {
        if range.start >= range.end || range.end > self.examples.len() {
            return Err(TrainingError("training range is invalid".into()));
        }
        Ok(TrainingView {
            schema: &self.schema,
            examples: &self.examples[range],
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChronologicalSplit {
    pub train_end_exclusive: usize,
    pub validation_end_exclusive: usize,
}

impl ChronologicalSplit {
    pub fn new(
        sample_count: usize,
        train_samples: usize,
        validation_samples: usize,
    ) -> Result<Self, TrainingError> {
        if train_samples == 0 || validation_samples == 0 {
            return Err(TrainingError(
                "chronological split requires nonempty train and validation sets".into(),
            ));
        }
        let validation_end_exclusive = train_samples
            .checked_add(validation_samples)
            .ok_or_else(|| TrainingError("chronological split overflows".into()))?;
        if validation_end_exclusive > sample_count {
            return Err(TrainingError(
                "chronological split exceeds dataset length".into(),
            ));
        }
        Ok(Self {
            train_end_exclusive: train_samples,
            validation_end_exclusive,
        })
    }

    pub fn train_range(self) -> Range<usize> {
        0..self.train_end_exclusive
    }

    pub fn validation_range(self) -> Range<usize> {
        self.train_end_exclusive..self.validation_end_exclusive
    }
}

#[derive(Clone, Debug)]
pub struct TrainingView<'a> {
    pub schema: &'a FeatureSchema,
    pub examples: &'a [TrainingExample],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WalkForwardPlan {
    pub folds: Vec<ChronologicalSplit>,
}

impl WalkForwardPlan {
    pub fn new(
        sample_count: usize,
        initial_train_samples: usize,
        validation_samples: usize,
        step_samples: usize,
    ) -> Result<Self, TrainingError> {
        if step_samples == 0 {
            return Err(TrainingError("walk-forward step must be positive".into()));
        }
        let mut folds = Vec::new();
        let mut train_end = initial_train_samples;
        while train_end
            .checked_add(validation_samples)
            .is_some_and(|end| end <= sample_count)
        {
            folds.push(ChronologicalSplit {
                train_end_exclusive: train_end,
                validation_end_exclusive: train_end + validation_samples,
            });
            train_end = train_end
                .checked_add(step_samples)
                .ok_or_else(|| TrainingError("walk-forward plan overflows".into()))?;
        }
        if folds.is_empty() {
            return Err(TrainingError(
                "walk-forward plan has no complete validation fold".into(),
            ));
        }
        Ok(Self { folds })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingError(pub String);

impl fmt::Display for TrainingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TrainingError {}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingReport {
    pub solver: String,
    pub model_vintage: String,
    pub feature_schema_sha256: String,
    pub dataset_sha256: String,
    pub train_samples: usize,
    pub validation_samples: usize,
    pub train_mse: f64,
    pub validation_mse: f64,
    pub validation_mae: f64,
    pub validation_r2: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TrainedLinearModel {
    pub model: LinearModel,
    pub report: TrainingReport,
}

#[derive(Clone, Debug)]
pub struct WalkForwardTraining {
    pub folds: Vec<TrainedLinearModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LinearSolver {
    OrdinaryLeastSquares,
    Ridge { alpha: f64 },
}

impl LinearSolver {
    #[cfg(feature = "smartcore-backend")]
    fn name(&self) -> String {
        match self {
            Self::OrdinaryLeastSquares => "ordinary_least_squares".into(),
            Self::Ridge { alpha } => format!("ridge(alpha={alpha})"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearTrainingConfig {
    pub model_vintage: String,
    pub model_type: String,
    pub solver: LinearSolver,
}

#[cfg(feature = "smartcore-backend")]
pub fn train_linear(
    dataset: &TrainingDataset,
    split: ChronologicalSplit,
    config: &LinearTrainingConfig,
) -> Result<TrainedLinearModel, TrainingError> {
    smartcore_backend::train_linear(dataset, split, config)
}

#[cfg(feature = "smartcore-backend")]
pub fn train_linear_walk_forward(
    dataset: &TrainingDataset,
    plan: &WalkForwardPlan,
    config: &LinearTrainingConfig,
) -> Result<WalkForwardTraining, TrainingError> {
    let folds = plan
        .folds
        .iter()
        .copied()
        .map(|split| train_linear(dataset, split, config))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(WalkForwardTraining { folds })
}

#[cfg(not(feature = "smartcore-backend"))]
pub fn train_linear_walk_forward(
    _dataset: &TrainingDataset,
    _plan: &WalkForwardPlan,
    _config: &LinearTrainingConfig,
) -> Result<WalkForwardTraining, TrainingError> {
    Err(TrainingError(
        "train_linear_walk_forward requires the smartcore-backend feature".into(),
    ))
}

#[cfg(not(feature = "smartcore-backend"))]
pub fn train_linear(
    _dataset: &TrainingDataset,
    _split: ChronologicalSplit,
    _config: &LinearTrainingConfig,
) -> Result<TrainedLinearModel, TrainingError> {
    Err(TrainingError(
        "train_linear requires the smartcore-backend feature".into(),
    ))
}

#[cfg(feature = "smartcore-backend")]
fn metrics(weights: &[f64], bias: f64, examples: &[TrainingExample]) -> (f64, f64, Option<f64>) {
    let predictions: Vec<f64> = examples
        .iter()
        .map(|example| {
            bias + weights
                .iter()
                .zip(&example.features.values)
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
        })
        .collect();
    let targets: Vec<f64> = examples.iter().map(|example| example.target).collect();
    let errors: Vec<f64> = predictions
        .iter()
        .zip(&targets)
        .map(|(prediction, target)| prediction - target)
        .collect();
    let mse = errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64;
    let mae = errors.iter().map(|error| error.abs()).sum::<f64>() / errors.len() as f64;
    let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let total_sum = targets
        .iter()
        .map(|target| (target - target_mean) * (target - target_mean))
        .sum::<f64>();
    let r2 = if total_sum > 0.0 {
        Some(1.0 - errors.iter().map(|error| error * error).sum::<f64>() / total_sum)
    } else {
        None
    };
    (mse, mae, r2)
}

#[cfg(feature = "smartcore-backend")]
mod smartcore_backend {
    use super::*;
    use smartcore::linalg::basic::arrays::Array;
    use smartcore::linalg::basic::matrix::DenseMatrix;
    use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};
    use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};

    pub(super) fn train_linear(
        dataset: &TrainingDataset,
        split: ChronologicalSplit,
        config: &LinearTrainingConfig,
    ) -> Result<TrainedLinearModel, TrainingError> {
        dataset.validate_split(split)?;
        let train = dataset.range(split.train_range())?;
        let validation = dataset.range(split.validation_range())?;
        let train_rows: Vec<Vec<f64>> = train
            .examples
            .iter()
            .map(|example| example.features.values.clone())
            .collect();
        let train_targets: Vec<f64> = train
            .examples
            .iter()
            .map(|example| example.target)
            .collect();
        let matrix = DenseMatrix::from_2d_vec(&train_rows)
            .map_err(|error| TrainingError(format!("smartcore matrix: {error}")))?;
        let (weights, bias) = match config.solver {
            LinearSolver::OrdinaryLeastSquares => {
                let model = LinearRegression::fit(
                    &matrix,
                    &train_targets,
                    LinearRegressionParameters::default(),
                )
                .map_err(|error| TrainingError(format!("smartcore linear regression: {error}")))?;
                (
                    model
                        .coefficients()
                        .iterator(0)
                        .copied()
                        .collect::<Vec<f64>>(),
                    *model.intercept(),
                )
            }
            LinearSolver::Ridge { alpha } => {
                if !alpha.is_finite() || alpha < 0.0 {
                    return Err(TrainingError(
                        "ridge alpha must be finite and nonnegative".into(),
                    ));
                }
                let parameters = RidgeRegressionParameters::default().with_alpha(alpha);
                let model =
                    RidgeRegression::fit(&matrix, &train_targets, parameters).map_err(|error| {
                        TrainingError(format!("smartcore ridge regression: {error}"))
                    })?;
                (
                    model
                        .coefficients()
                        .iterator(0)
                        .copied()
                        .collect::<Vec<f64>>(),
                    *model.intercept(),
                )
            }
        };
        let model = LinearModel::new(
            ModelMetadata::new(&config.model_vintage, &config.model_type),
            dataset.schema.clone(),
            weights.clone(),
            bias,
        )
        .map_err(|error: ModelError| TrainingError(error.to_string()))?;
        let (train_mse, _, _) = metrics(&weights, bias, train.examples);
        let (validation_mse, validation_mae, validation_r2) =
            metrics(&weights, bias, validation.examples);
        Ok(TrainedLinearModel {
            model,
            report: TrainingReport {
                solver: config.solver.name(),
                model_vintage: config.model_vintage.clone(),
                feature_schema_sha256: dataset.schema.sha256.clone(),
                dataset_sha256: dataset.sha256()?,
                train_samples: train.examples.len(),
                validation_samples: validation.examples.len(),
                train_mse,
                validation_mse,
                validation_mae,
                validation_r2,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_ml::FeatureSchema;

    #[cfg(feature = "smartcore-backend")]
    fn dataset() -> TrainingDataset {
        let schema = FeatureSchema::new(vec!["x".into()]).expect("schema");
        let examples = (0..8)
            .map(|index| {
                let x = index as f64;
                let features = FeatureVector::new(schema.clone(), vec![x]).expect("features");
                TrainingExample::new(index, index, index + 1, features, 2.0 * x + 1.0)
                    .expect("example")
            })
            .collect();
        TrainingDataset::new(schema, examples).expect("dataset")
    }

    #[test]
    fn rejects_future_features_and_non_chronological_rows() {
        let schema = FeatureSchema::new(vec!["x".into()]).expect("schema");
        let features = FeatureVector::new(schema.clone(), vec![1.0]).expect("features");
        let error = TrainingExample::new(2, 1, 3, features, 1.0).expect_err("future feature");
        assert!(error.to_string().contains("feature_end_ns"));
    }

    #[test]
    fn walk_forward_plan_is_chronological_and_expanding() {
        let plan = WalkForwardPlan::new(10, 4, 2, 2).expect("walk-forward plan");
        assert_eq!(
            plan.folds,
            vec![
                ChronologicalSplit {
                    train_end_exclusive: 4,
                    validation_end_exclusive: 6,
                },
                ChronologicalSplit {
                    train_end_exclusive: 6,
                    validation_end_exclusive: 8,
                },
                ChronologicalSplit {
                    train_end_exclusive: 8,
                    validation_end_exclusive: 10,
                },
            ]
        );
    }

    #[test]
    fn chronological_split_never_shuffles_rows() {
        let split = ChronologicalSplit::new(8, 5, 3).expect("split");
        assert_eq!(split.train_range(), 0..5);
        assert_eq!(split.validation_range(), 5..8);
    }

    #[test]
    fn rejects_a_training_label_that_crosses_the_validation_boundary() {
        let schema = FeatureSchema::new(vec!["x".into()]).expect("schema");
        let examples = (0..4)
            .map(|index| {
                let features =
                    FeatureVector::new(schema.clone(), vec![index as f64]).expect("features");
                let label_end = if index == 1 { 3 } else { index + 1 };
                TrainingExample::new(index, index, label_end, features, index as f64)
                    .expect("example")
            })
            .collect();
        let dataset = TrainingDataset::new(schema, examples).expect("dataset");
        let error = dataset
            .validate_split(ChronologicalSplit::new(4, 2, 2).expect("split"))
            .expect_err("label boundary leak");
        assert!(error.to_string().contains("crosses the validation"));
    }

    #[cfg(feature = "smartcore-backend")]
    #[test]
    fn smartcore_linear_training_exports_bullet_model_and_validation_metrics() {
        let trained = train_linear(
            &dataset(),
            ChronologicalSplit::new(8, 6, 2).expect("split"),
            &LinearTrainingConfig {
                model_vintage: "linear-test-v1".into(),
                model_type: "ols".into(),
                solver: LinearSolver::OrdinaryLeastSquares,
            },
        )
        .expect("training");
        assert_eq!(trained.model.weights, vec![2.0]);
        assert!((trained.model.bias - 1.0).abs() < 1e-10);
        assert!(trained.report.validation_mse < 1e-18);
        assert_eq!(trained.report.validation_samples, 2);
    }
}

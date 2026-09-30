//! Backend-neutral, timestamp-based sliding training windows.
use crate::{TrainingDataset, TrainingError};
use bullet_ml::{
    FeatureSchema, FeatureTransform, FeatureVector, Model, ModelWindow, PreprocessedModel,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeFold {
    pub train_start_ns: u64,
    pub fit_asof_ns: u64,
    pub label_cutoff_ns: u64,
    pub predict_start_ns: u64,
    pub predict_end_ns: u64,
}

impl TimeFold {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.train_start_ns >= self.fit_asof_ns
            || self.label_cutoff_ns > self.fit_asof_ns
            || self.fit_asof_ns > self.predict_start_ns
            || self.predict_start_ns >= self.predict_end_ns
        {
            return Err(TrainingError("invalid rolling time fold".into()));
        }
        Ok(())
    }
    pub fn training_indices(&self, dataset: &TrainingDataset) -> Result<Vec<usize>, TrainingError> {
        self.validate()?;
        // Select every label independently: variable horizons need not mature in row order.
        let first = dataset
            .examples()
            .partition_point(|e| e.decision_time_ns < self.train_start_ns);
        let end = dataset
            .examples()
            .partition_point(|e| e.decision_time_ns < self.fit_asof_ns);
        Ok((first..end)
            .filter(|&i| dataset.examples()[i].label_end_ns <= self.label_cutoff_ns)
            .collect())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollingPlan {
    folds: Vec<TimeFold>,
}

impl RollingPlan {
    pub fn new(
        first_fit_ns: u64,
        end_ns: u64,
        history_ns: u64,
        step_ns: u64,
        label_gap_ns: u64,
        activation_delay_ns: u64,
    ) -> Result<Self, TrainingError> {
        if history_ns == 0 || step_ns == 0 || activation_delay_ns >= step_ns {
            return Err(TrainingError(
                "history/step must be positive and activation delay smaller than step".into(),
            ));
        }
        let mut folds = Vec::new();
        let mut fit = first_fit_ns;
        loop {
            let start = fit
                .checked_add(activation_delay_ns)
                .ok_or_else(|| TrainingError("rolling time overflow".into()))?;
            if start >= end_ns {
                break;
            }
            let end = start
                .checked_add(step_ns)
                .ok_or_else(|| TrainingError("rolling time overflow".into()))?
                .min(end_ns);
            folds.push(TimeFold {
                train_start_ns: fit
                    .checked_sub(history_ns)
                    .ok_or_else(|| TrainingError("training window precedes epoch".into()))?,
                fit_asof_ns: fit,
                label_cutoff_ns: fit
                    .checked_sub(label_gap_ns)
                    .ok_or_else(|| TrainingError("label gap precedes epoch".into()))?,
                predict_start_ns: start,
                predict_end_ns: end,
            });
            if end == end_ns {
                break;
            }
            fit = fit
                .checked_add(step_ns)
                .ok_or_else(|| TrainingError("rolling time overflow".into()))?;
        }
        Self::from_folds(folds)
    }
    pub fn from_folds(folds: Vec<TimeFold>) -> Result<Self, TrainingError> {
        let plan = Self { folds };
        plan.validate()?;
        Ok(plan)
    }
    pub fn folds(&self) -> &[TimeFold] {
        &self.folds
    }
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.folds.is_empty() {
            return Err(TrainingError("rolling plan has no folds".into()));
        }
        for fold in &self.folds {
            fold.validate()?;
        }
        if self.folds.windows(2).any(|w| {
            w[0].predict_end_ns != w[1].predict_start_ns || w[0].fit_asof_ns >= w[1].fit_asof_ns
        }) {
            return Err(TrainingError(
                "rolling folds must advance and cover a contiguous prediction interval".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Scaling {
    None,
    Standardize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightNormalization {
    Preserve,
    TrainingMeanOne,
}

pub struct RollingOptions {
    pub minimum_training_samples: usize,
    pub scaling: Scaling,
    pub weight_normalization: WeightNormalization,
    pub vintage_prefix: String,
}

pub struct FitBatch<'a> {
    pub schema: &'a FeatureSchema,
    pub features: &'a [Vec<f64>],
    pub targets: &'a [f64],
    pub weights: &'a [f64],
    pub model_vintage: &'a str,
}

impl FitBatch<'_> {
    pub fn validate(&self) -> Result<(), TrainingError> {
        self.schema
            .validate()
            .map_err(|e| TrainingError(e.to_string()))?;
        if self.features.is_empty()
            || self.features.len() != self.targets.len()
            || self.features.len() != self.weights.len()
            || self.model_vintage.trim().is_empty()
        {
            return Err(TrainingError(
                "invalid fit batch dimensions or vintage".into(),
            ));
        }
        if self.targets.iter().any(|v| !v.is_finite())
            || self.weights.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self.features.iter().any(|row| {
                row.len() != self.schema.names.len() || row.iter().any(|v| !v.is_finite())
            })
        {
            return Err(TrainingError(
                "fit batch requires finite features/targets and positive finite weights".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitReceipt {
    pub backend: String,
    pub backend_version: String,
    pub parameters: Value,
    pub model_sha256: String,
}

pub trait ArtifactModel: Model {
    fn artifact_sha256(&self) -> Result<String, TrainingError>;
}

impl ArtifactModel for bullet_ml::LinearModel {
    fn artifact_sha256(&self) -> Result<String, TrainingError> {
        hash(self)
    }
}

impl ArtifactModel for bullet_ml::LogisticRegressionModel {
    fn artifact_sha256(&self) -> Result<String, TrainingError> {
        self.validate().map_err(|e| TrainingError(e.to_string()))?;
        hash(self)
    }
}
impl ArtifactModel for bullet_ml::HuberRegressionModel {
    fn artifact_sha256(&self) -> Result<String, TrainingError> {
        self.validate().map_err(|e| TrainingError(e.to_string()))?;
        hash(self)
    }
}

pub trait RollingTrainer {
    type Model: ArtifactModel;
    fn fit(&mut self, batch: &FitBatch<'_>) -> Result<(Self::Model, FitReceipt), TrainingError>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FoldReport {
    pub fold: TimeFold,
    pub model_vintage: String,
    pub feature_schema_sha256: String,
    pub training_samples: usize,
    pub training_indices_sha256: String,
    pub training_dataset_sha256: String,
    pub sample_weights_sha256: String,
    pub raw_sample_weights_sha256: String,
    pub weight_normalization: WeightNormalization,
    pub transformed_features_sha256: String,
    pub maximum_training_label_end_ns: u64,
    pub transform: FeatureTransform,
    pub receipt: FitReceipt,
    pub weighted_train_mse: f64,
    pub bundle_sha256: String,
}

impl FoldReport {
    pub fn validate(&self) -> Result<(), TrainingError> {
        self.fold.validate()?;
        if self.training_samples == 0
            || self.maximum_training_label_end_ns > self.fold.label_cutoff_ns
            || !self.weighted_train_mse.is_finite()
        {
            return Err(TrainingError(
                "invalid fold report causal/metric contract".into(),
            ));
        }
        let mut payload = self.clone();
        payload.bundle_sha256.clear();
        if hash(&payload)? != self.bundle_sha256 {
            return Err(TrainingError("fold report hash mismatch".into()));
        }
        Ok(())
    }
}

pub struct FittedFold<M> {
    pub model: PreprocessedModel<M>,
    pub report: FoldReport,
}

impl<M: ArtifactModel> FittedFold<M> {
    pub fn into_window(self) -> Result<ModelWindow<PreprocessedModel<M>>, TrainingError> {
        self.report.validate()?;
        if self.model.model().artifact_sha256()? != self.report.receipt.model_sha256 {
            return Err(TrainingError("model/report artifact hash mismatch".into()));
        }
        if self.model.transform() != &self.report.transform
            || self.model.feature_schema().sha256 != self.report.feature_schema_sha256
        {
            return Err(TrainingError(
                "model/report transform or schema mismatch".into(),
            ));
        }
        if self.report.maximum_training_label_end_ns > self.report.fold.label_cutoff_ns
            || self.model.metadata().model_vintage != self.report.model_vintage
        {
            return Err(TrainingError(
                "model/report causal contract mismatch".into(),
            ));
        }
        ModelWindow::new(
            self.report.fold.fit_asof_ns,
            self.report.fold.predict_start_ns,
            self.report.fold.predict_end_ns,
            self.model,
        )
        .and_then(|window| window.with_training_report_sha256(self.report.bundle_sha256))
        .map_err(|e| TrainingError(e.to_string()))
    }
}

pub fn train_rolling<B: RollingTrainer>(
    dataset: &TrainingDataset,
    weights: &[f64],
    plan: &RollingPlan,
    options: &RollingOptions,
    backend: &mut B,
) -> Result<Vec<FittedFold<B::Model>>, TrainingError> {
    plan.validate()?;
    if weights.len() != dataset.len() || weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
        return Err(TrainingError(
            "one positive finite sample weight is required per row".into(),
        ));
    }
    if options.minimum_training_samples == 0 || options.vintage_prefix.trim().is_empty() {
        return Err(TrainingError(
            "minimum training samples and vintage prefix are required".into(),
        ));
    }
    let mut fitted = Vec::new();
    for fold in plan.folds() {
        let indices = fold.training_indices(dataset)?;
        if indices.len() < options.minimum_training_samples {
            return Err(TrainingError(format!(
                "insufficient mature training samples at {}: {}",
                fold.fit_asof_ns,
                indices.len()
            )));
        }
        let rows: Vec<_> = indices.iter().map(|&i| &dataset.examples()[i]).collect();
        let mut selected_weights: Vec<_> = indices.iter().map(|&i| weights[i]).collect();
        let raw_sample_weights_sha256 = hash(&selected_weights)?;
        if options.weight_normalization == WeightNormalization::TrainingMeanOne {
            let mean = selected_weights.iter().sum::<f64>() / selected_weights.len() as f64;
            if !mean.is_finite() {
                return Err(TrainingError("non-finite training weight mean".into()));
            }
            for weight in &mut selected_weights {
                *weight /= mean;
            }
        }
        let raw_features: Vec<_> = rows.iter().map(|r| r.features.values.clone()).collect();
        let transform = fit_transform(&raw_features, options.scaling)?;
        let features: Vec<_> = rows
            .iter()
            .map(|r| {
                transform
                    .apply(&r.features)
                    .map(|v| v.values)
                    .map_err(|e| TrainingError(e.to_string()))
            })
            .collect::<Result<_, _>>()?;
        let targets: Vec<_> = rows.iter().map(|r| r.target).collect();
        let vintage = format!("{}@{}", options.vintage_prefix, fold.fit_asof_ns);
        let batch = FitBatch {
            schema: dataset.schema(),
            features: &features,
            targets: &targets,
            weights: &selected_weights,
            model_vintage: &vintage,
        };
        batch.validate()?;
        let (mut model, receipt) = backend.fit(&batch)?;
        if model.artifact_sha256()? != receipt.model_sha256 {
            return Err(TrainingError(
                "backend model/receipt artifact hash mismatch".into(),
            ));
        }
        if model.metadata().model_vintage != vintage || model.feature_schema() != dataset.schema() {
            return Err(TrainingError(
                "backend changed model vintage or feature schema".into(),
            ));
        }
        let mut sum_error = 0.0;
        for ((values, target), weight) in features.iter().zip(&targets).zip(&selected_weights) {
            let vector = FeatureVector::new(dataset.schema().clone(), values.clone())
                .map_err(|e| TrainingError(e.to_string()))?;
            let score = model
                .predict(&vector)
                .map_err(|e| TrainingError(e.to_string()))?
                .score;
            sum_error += weight * (score - target).powi(2);
        }
        let mse = sum_error / selected_weights.iter().sum::<f64>();
        if !mse.is_finite() {
            return Err(TrainingError("non-finite training metric".into()));
        }
        let mut report = FoldReport {
            fold: fold.clone(),
            model_vintage: vintage,
            feature_schema_sha256: dataset.schema().sha256.clone(),
            training_samples: rows.len(),
            training_indices_sha256: hash(&indices)?,
            training_dataset_sha256: hash(&(dataset.schema(), &rows))?,
            sample_weights_sha256: hash(&selected_weights)?,
            raw_sample_weights_sha256,
            weight_normalization: options.weight_normalization,
            transformed_features_sha256: hash(&features)?,
            maximum_training_label_end_ns: rows
                .iter()
                .map(|r| r.label_end_ns)
                .max()
                .expect("nonempty training rows"),
            transform: transform.clone(),
            receipt,
            weighted_train_mse: mse,
            bundle_sha256: String::new(),
        };
        report.bundle_sha256 = hash(&report)?;
        fitted.push(FittedFold {
            model: PreprocessedModel::new(model, transform)
                .map_err(|e| TrainingError(e.to_string()))?,
            report,
        });
    }
    Ok(fitted)
}

fn fit_transform(rows: &[Vec<f64>], scaling: Scaling) -> Result<FeatureTransform, TrainingError> {
    if matches!(scaling, Scaling::None) {
        return Ok(FeatureTransform::Identity);
    }
    let dimensions = rows[0].len();
    let mut mean = vec![0.0; dimensions];
    for row in rows {
        for (m, value) in mean.iter_mut().zip(row) {
            *m += value;
        }
    }
    for m in &mut mean {
        *m /= rows.len() as f64;
    }
    let mut scale = vec![0.0; dimensions];
    for row in rows {
        for ((s, value), m) in scale.iter_mut().zip(row).zip(&mean) {
            *s += (value - m).powi(2);
        }
    }
    for s in &mut scale {
        *s = (*s / rows.len() as f64).sqrt();
        if *s == 0.0 {
            *s = 1.0;
        }
    }
    let transform = FeatureTransform::Standardize { mean, scale };
    transform
        .validate(dimensions)
        .map_err(|e| TrainingError(e.to_string()))?;
    Ok(transform)
}

pub(crate) fn hash<T: Serialize + ?Sized>(value: &T) -> Result<String, TrainingError> {
    let bytes = serde_json::to_vec(value).map_err(|e| TrainingError(e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

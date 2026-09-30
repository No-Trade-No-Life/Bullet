//! Optional native XGBoost backend through the community `xgb` wrapper.
//! No Python process, interpreter, or Python serialization is used.
use crate::{
    TrainingError,
    rolling::{ArtifactModel, FitBatch, FitReceipt, RollingTrainer, hash},
};
use bullet_ml::{FeatureSchema, FeatureVector, Model, ModelMetadata, Prediction};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use xgb::{Booster, DMatrix, parameters::BoosterParameters};

pub const NATIVE_VERSION: [u32; 3] = [3, 2, 0];
pub const WRAPPER_VERSION: &str = "3.0.6";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    BinaryLogistic,
    SquaredError,
}
impl Objective {
    fn native_name(self) -> &'static str {
        match self {
            Self::BinaryLogistic => "binary:logistic",
            Self::SquaredError => "reg:squarederror",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XgboostConfig {
    pub objective: Objective,
    pub rounds: u32,
    pub max_depth: u32,
    pub learning_rate: f64,
    pub min_child_weight: f64,
    pub max_bin: u32,
    pub reg_lambda: f64,
    pub reg_alpha: f64,
    pub subsample: f64,
    pub colsample_bytree: f64,
    pub seed: u32,
    pub minimum_rows_per_class: usize,
}

impl Default for XgboostConfig {
    fn default() -> Self {
        Self {
            objective: Objective::SquaredError,
            rounds: 32,
            max_depth: 3,
            learning_rate: 0.1,
            min_child_weight: 1.0,
            max_bin: 64,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            seed: 0,
            minimum_rows_per_class: 1,
        }
    }
}

impl XgboostConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.rounds == 0
            || self.rounds > i32::MAX as u32
            || self.max_depth == 0
            || self.max_bin < 2
            || self.minimum_rows_per_class == 0
        {
            return Err(TrainingError(
                "invalid XGBoost rounds/depth/bins/class minimum".into(),
            ));
        }
        if [self.learning_rate, self.subsample, self.colsample_bytree]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0 || *v > 1.0)
            || [self.min_child_weight, self.reg_lambda, self.reg_alpha]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(TrainingError("invalid XGBoost numeric parameters".into()));
        }
        Ok(())
    }
    fn parameters(&self) -> Vec<(&'static str, String)> {
        vec![
            ("objective", self.objective.native_name().into()),
            ("booster", "gbtree".into()),
            ("device", "cpu".into()),
            ("nthread", "1".into()),
            ("tree_method", "hist".into()),
            ("verbosity", "0".into()),
            ("validate_parameters", "1".into()),
            // Override the wrapper's fixed 0.5 initialization explicitly.
            ("boost_from_average", "1".into()),
            ("max_depth", self.max_depth.to_string()),
            ("eta", self.learning_rate.to_string()),
            ("min_child_weight", self.min_child_weight.to_string()),
            ("max_bin", self.max_bin.to_string()),
            ("lambda", self.reg_lambda.to_string()),
            ("alpha", self.reg_alpha.to_string()),
            ("subsample", self.subsample.to_string()),
            ("colsample_bytree", self.colsample_bytree.to_string()),
            ("seed", self.seed.to_string()),
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XgboostArtifact {
    format_version: u32,
    metadata: ModelMetadata,
    feature_schema: FeatureSchema,
    config: XgboostConfig,
    native_version: [u32; 3],
    wrapper_version: String,
    model_json: String,
    model_json_sha256: String,
}

impl XgboostArtifact {
    pub fn sha256(&self) -> Result<String, TrainingError> {
        hash(self)
    }
    pub fn native_version(&self) -> [u32; 3] {
        self.native_version
    }
    fn validate(&self) -> Result<(), TrainingError> {
        self.feature_schema.validate().map_err(error)?;
        self.config.validate()?;
        if self.format_version != 1
            || self.native_version != NATIVE_VERSION
            || self.wrapper_version != WRAPPER_VERSION
            || self.metadata.model_vintage.trim().is_empty()
            || self.metadata.model_type != self.config.objective.native_name()
            || digest(self.model_json.as_bytes()) != self.model_json_sha256
        {
            return Err(TrainingError(
                "XGBoost artifact metadata/version/hash mismatch".into(),
            ));
        }
        let model: Value = serde_json::from_str(&self.model_json).map_err(error)?;
        check_version(&model)?;
        let learner = &model["learner"];
        let parameters = &learner["learner_model_param"];
        if learner["objective"]["name"].as_str() != Some(self.config.objective.native_name())
            || learner["gradient_booster"]["name"] != "gbtree"
            || parameters["num_feature"]
                .as_str()
                .and_then(|s| s.parse::<usize>().ok())
                != Some(self.feature_schema.names.len())
            || parameters["num_class"].as_str() != Some("0")
            || parameters["num_target"].as_str() != Some("1")
        {
            return Err(TrainingError(
                "XGBoost artifact objective/feature/output shape mismatch".into(),
            ));
        }
        Ok(())
    }
}

pub struct XgboostModel {
    booster: Booster,
    artifact: XgboostArtifact,
    artifact_sha256: String,
}

impl XgboostModel {
    pub fn load(artifact: XgboostArtifact) -> Result<Self, TrainingError> {
        artifact.validate()?;
        let mut booster = Booster::load_buffer(artifact.model_json.as_bytes()).map_err(error)?;
        booster.set_param("nthread", "1").map_err(error)?;
        booster.set_param("device", "cpu").map_err(error)?;
        // A separate probe checks the linked library instead of trusting the artifact's version.
        verify_runtime_version()?;
        let artifact_sha256 = artifact.sha256()?;
        Ok(Self {
            booster,
            artifact,
            artifact_sha256,
        })
    }
    pub fn artifact(&self) -> &XgboostArtifact {
        &self.artifact
    }
    pub fn predict_rows(&self, features: &[FeatureVector]) -> Result<Vec<f64>, TrainingError> {
        if features.is_empty() {
            return Err(TrainingError("empty XGBoost prediction batch".into()));
        }
        let mut values = Vec::new();
        for row in features {
            row.validate().map_err(error)?;
            if row.schema != self.artifact.feature_schema {
                return Err(TrainingError("XGBoost feature schema mismatch".into()));
            }
            values.extend(
                row.values
                    .iter()
                    .map(|&v| float32(v))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        let matrix = DMatrix::from_dense(&values, features.len()).map_err(error)?;
        let scores = self.booster.predict(&matrix).map_err(error)?;
        if scores.len() != features.len() || scores.iter().any(|s| !s.is_finite()) {
            return Err(TrainingError(
                "XGBoost prediction shape/value mismatch".into(),
            ));
        }
        Ok(scores.into_iter().map(f64::from).collect())
    }
}

impl ArtifactModel for XgboostModel {
    fn artifact_sha256(&self) -> Result<String, TrainingError> {
        Ok(self.artifact_sha256.clone())
    }
}

impl Model for XgboostModel {
    type Error = TrainingError;
    fn metadata(&self) -> &ModelMetadata {
        &self.artifact.metadata
    }
    fn feature_schema(&self) -> &FeatureSchema {
        &self.artifact.feature_schema
    }
    fn predict(&mut self, features: &FeatureVector) -> Result<Prediction, Self::Error> {
        let score = self.predict_rows(std::slice::from_ref(features))?[0];
        Prediction::score(score).map(|p| p.diagnostic(json!({"model_sha256": self.artifact_sha256, "native_version": NATIVE_VERSION, "wrapper_version": WRAPPER_VERSION}))).map_err(error)
    }
}

pub struct XgboostTrainer {
    config: XgboostConfig,
}
impl XgboostTrainer {
    pub fn new(config: XgboostConfig) -> Result<Self, TrainingError> {
        config.validate()?;
        Ok(Self { config })
    }
}

impl RollingTrainer for XgboostTrainer {
    type Model = XgboostModel;
    fn fit(&mut self, batch: &FitBatch<'_>) -> Result<(Self::Model, FitReceipt), TrainingError> {
        batch.validate()?;
        if self.config.objective == Objective::BinaryLogistic {
            let mut counts = [0usize; 2];
            for &y in batch.targets {
                if y != 0.0 && y != 1.0 {
                    return Err(TrainingError(
                        "binary classification labels must be 0 or 1".into(),
                    ));
                }
                counts[y as usize] += 1;
            }
            if counts
                .iter()
                .any(|n| *n < self.config.minimum_rows_per_class)
            {
                return Err(TrainingError("insufficient rows per class".into()));
            }
        }
        let values = batch
            .features
            .iter()
            .flatten()
            .map(|&v| float32(v))
            .collect::<Result<Vec<_>, _>>()?;
        let labels = batch
            .targets
            .iter()
            .map(|&v| float32(v))
            .collect::<Result<Vec<_>, _>>()?;
        let weights = batch
            .weights
            .iter()
            .map(|&v| float32(v))
            .collect::<Result<Vec<_>, _>>()?;
        if weights.iter().any(|w| *w <= 0.0) {
            return Err(TrainingError(
                "sample weight underflow in float32 conversion".into(),
            ));
        }
        let mut matrix = DMatrix::from_dense(&values, batch.features.len()).map_err(error)?;
        matrix.set_labels(&labels).map_err(error)?;
        matrix.set_weights(&weights).map_err(error)?;
        let mut booster = Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&matrix])
            .map_err(error)?;
        for (name, value) in self.config.parameters() {
            booster.set_param(name, &value).map_err(error)?;
        }
        verify_runtime_version()?;
        for round in 0..self.config.rounds {
            booster.update(&matrix, round as i32).map_err(error)?;
        }
        let model_json = export_json(&booster)?;
        let artifact = XgboostArtifact {
            format_version: 1,
            metadata: ModelMetadata::new(batch.model_vintage, self.config.objective.native_name()),
            feature_schema: batch.schema.clone(),
            config: self.config.clone(),
            native_version: NATIVE_VERSION,
            wrapper_version: WRAPPER_VERSION.into(),
            model_json_sha256: digest(model_json.as_bytes()),
            model_json,
        };
        artifact.validate()?;
        let artifact_sha256 = artifact.sha256()?;
        let receipt = FitReceipt {
            backend: "xgboost-cpu-hist".into(),
            backend_version: "3.2.0".into(),
            parameters: json!({"config": self.config, "native_parameters": self.config.parameters(), "wrapper_version": WRAPPER_VERSION, "matrix_dtype": "float32", "initial_intercept": "estimated_from_training_data"}),
            model_sha256: artifact_sha256.clone(),
        };
        Ok((
            XgboostModel {
                booster,
                artifact,
                artifact_sha256,
            },
            receipt,
        ))
    }
}

fn verify_runtime_version() -> Result<(), TrainingError> {
    let mut probe = Booster::new(&BoosterParameters::default()).map_err(error)?;
    probe.set_param("num_feature", "1").map_err(error)?;
    probe
        .set_param("objective", "reg:squarederror")
        .map_err(error)?;
    check_version(&serde_json::from_str(&export_json(&probe)?).map_err(error)?)
}

fn export_json(booster: &Booster) -> Result<String, TrainingError> {
    // COMPATIBILITY: xgb 3.0.6 save_buffer passes non-NUL-terminated config to C.
    // Bullet's native adapter uses the CString-backed file API until upstream fixes
    // that entry point and artifact roundtrip tests pass with the replacement version.
    let directory = tempfile::tempdir().map_err(error)?;
    let path = directory.path().join("model.json");
    booster.save(&path).map_err(error)?;
    std::fs::read_to_string(path).map_err(error)
}
fn check_version(model: &Value) -> Result<(), TrainingError> {
    if model["version"] != json!(NATIVE_VERSION) {
        return Err(TrainingError(format!(
            "native XGBoost 3.2.0 required; found {}",
            model["version"]
        )));
    }
    Ok(())
}
fn float32(value: f64) -> Result<f32, TrainingError> {
    let result = value as f32;
    if !result.is_finite() {
        return Err(TrainingError(
            "non-finite/overflowing XGBoost float32 input".into(),
        ));
    }
    Ok(result)
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn error(error: impl std::fmt::Display) -> TrainingError {
    TrainingError(error.to_string())
}

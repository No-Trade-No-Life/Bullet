//! Serializable, training-fitted feature transforms shared by all backends.
use crate::{FeatureSchema, FeatureVector, Model, ModelError, ModelMetadata, Prediction};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FeatureTransform {
    Identity,
    Standardize { mean: Vec<f64>, scale: Vec<f64> },
}

impl FeatureTransform {
    pub fn validate(&self, dimensions: usize) -> Result<(), ModelError> {
        if let Self::Standardize { mean, scale } = self
            && (mean.len() != dimensions
                || scale.len() != dimensions
                || mean.iter().any(|v| !v.is_finite())
                || scale.iter().any(|v| !v.is_finite() || *v <= 0.0))
        {
            return Err(ModelError("invalid standardization parameters".into()));
        }
        Ok(())
    }

    pub fn apply(&self, features: &FeatureVector) -> Result<FeatureVector, ModelError> {
        features.validate().map_err(|e| ModelError(e.to_string()))?;
        self.validate(features.values.len())?;
        let values = match self {
            Self::Identity => features.values.clone(),
            Self::Standardize { mean, scale } => features
                .values
                .iter()
                .zip(mean)
                .zip(scale)
                .map(|((x, m), s)| (x - m) / s)
                .collect(),
        };
        FeatureVector::new(features.schema.clone(), values).map_err(|e| ModelError(e.to_string()))
    }
}

pub struct PreprocessedModel<M> {
    model: M,
    transform: FeatureTransform,
}

impl<M: Model> PreprocessedModel<M> {
    pub fn new(model: M, transform: FeatureTransform) -> Result<Self, ModelError> {
        model
            .feature_schema()
            .validate()
            .map_err(|e| ModelError(e.to_string()))?;
        transform.validate(model.feature_schema().names.len())?;
        Ok(Self { model, transform })
    }
    pub fn model(&self) -> &M {
        &self.model
    }
    pub fn transform(&self) -> &FeatureTransform {
        &self.transform
    }
}

impl<M: Model> Model for PreprocessedModel<M> {
    type Error = ModelError;
    fn metadata(&self) -> &ModelMetadata {
        self.model.metadata()
    }
    fn feature_schema(&self) -> &FeatureSchema {
        self.model.feature_schema()
    }
    fn predict(&mut self, features: &FeatureVector) -> Result<Prediction, Self::Error> {
        if &features.schema != self.feature_schema() {
            return Err(ModelError(
                "preprocessed model feature schema mismatch".into(),
            ));
        }
        self.model
            .predict(&self.transform.apply(features)?)
            .map_err(|e| ModelError(e.to_string()))
    }
}

//! Native-free inference artifacts for fitted linear classification/robust regression.
use crate::{
    FeatureSchema, FeatureVector, LinearModel, Model, ModelError, ModelMetadata, Prediction,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Numerically stable probability for binary class 1.
pub fn logistic_probability(margin: f64) -> f64 {
    if margin >= 0.0 {
        1.0 / (1.0 + (-margin).exp())
    } else {
        let e = margin.exp();
        e / (1.0 + e)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogisticRegressionModel {
    linear: LinearModel,
    classes: [u8; 2],
}
impl LogisticRegressionModel {
    pub fn new(linear: LinearModel) -> Result<Self, ModelError> {
        let model = Self {
            linear,
            classes: [0, 1],
        };
        model.validate()?;
        Ok(model)
    }
    pub fn linear(&self) -> &LinearModel {
        &self.linear
    }
    pub fn validate(&self) -> Result<(), ModelError> {
        self.linear.validate()?;
        if self.classes != [0, 1]
            || self.linear.metadata.model_type != "linear-logistic-regression"
            || self.linear.metadata.model_vintage.trim().is_empty()
        {
            return Err(ModelError(
                "invalid binary logistic model metadata/classes".into(),
            ));
        }
        Ok(())
    }
}
impl Model for LogisticRegressionModel {
    type Error = ModelError;
    fn metadata(&self) -> &ModelMetadata {
        &self.linear.metadata
    }
    fn feature_schema(&self) -> &FeatureSchema {
        &self.linear.feature_schema
    }
    fn predict(&mut self, features: &FeatureVector) -> Result<Prediction, Self::Error> {
        self.validate()?;
        let margin = self.linear.predict(features)?.score;
        Ok(Prediction::score(logistic_probability(margin))?
            .diagnostic(json!({"margin": margin, "classes": self.classes, "score_class": 1})))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HuberRegressionModel {
    linear: LinearModel,
    scale: f64,
}
impl HuberRegressionModel {
    pub fn new(linear: LinearModel, scale: f64) -> Result<Self, ModelError> {
        let model = Self { linear, scale };
        model.validate()?;
        Ok(model)
    }
    pub fn linear(&self) -> &LinearModel {
        &self.linear
    }
    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn validate(&self) -> Result<(), ModelError> {
        self.linear.validate()?;
        if !self.scale.is_finite()
            || self.scale <= 0.0
            || self.linear.metadata.model_type != "huber-regression"
            || self.linear.metadata.model_vintage.trim().is_empty()
        {
            return Err(ModelError("invalid Huber model scale/metadata".into()));
        }
        Ok(())
    }
}
impl Model for HuberRegressionModel {
    type Error = ModelError;
    fn metadata(&self) -> &ModelMetadata {
        &self.linear.metadata
    }
    fn feature_schema(&self) -> &FeatureSchema {
        &self.linear.feature_schema
    }
    fn predict(&mut self, features: &FeatureVector) -> Result<Prediction, Self::Error> {
        self.validate()?;
        Ok(self
            .linear
            .predict(features)?
            .diagnostic(json!({"fitted_scale": self.scale})))
    }
}

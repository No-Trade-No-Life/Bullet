//! Decision-time model routing. Pipeline and mapper state survive model changes.
use crate::{FeatureOutput, FeaturePipeline, Model, ModelError, TargetMapper, WarmupPolicy};
use bullet_strategy::{DecisionContext, Strategy, TargetIntent};
use serde_json::json;
use std::collections::BTreeSet;

pub struct ModelWindow<M> {
    fit_asof_ns: u64,
    valid_from_ns: u64,
    valid_until_ns: u64,
    model: M,
    training_report_sha256: Option<String>,
}

impl<M: Model> ModelWindow<M> {
    pub fn with_training_report_sha256(mut self, sha256: String) -> Result<Self, ModelError> {
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(ModelError("invalid training report hash".into()));
        }
        self.training_report_sha256 = Some(sha256);
        Ok(self)
    }

    pub fn new(
        fit_asof_ns: u64,
        valid_from_ns: u64,
        valid_until_ns: u64,
        model: M,
    ) -> Result<Self, ModelError> {
        if fit_asof_ns > valid_from_ns || valid_from_ns >= valid_until_ns {
            return Err(ModelError("invalid model availability interval".into()));
        }
        if model.metadata().model_vintage.trim().is_empty() {
            return Err(ModelError("model vintage must not be empty".into()));
        }
        Ok(Self {
            fit_asof_ns,
            valid_from_ns,
            valid_until_ns,
            model,
            training_report_sha256: None,
        })
    }
}

pub struct ScheduledMlStrategy<P, M, T> {
    pipeline: P,
    windows: Vec<ModelWindow<M>>,
    mapper: T,
    warmup: WarmupPolicy,
    cursor: usize,
    previous_time_ns: Option<u64>,
}

impl<P: FeaturePipeline, M: Model, T: TargetMapper> ScheduledMlStrategy<P, M, T> {
    pub fn new(
        pipeline: P,
        windows: Vec<ModelWindow<M>>,
        mapper: T,
        warmup: WarmupPolicy,
    ) -> Result<Self, ModelError> {
        pipeline
            .schema()
            .validate()
            .map_err(|e| ModelError(e.to_string()))?;
        if windows.is_empty() {
            return Err(ModelError("model schedule is empty".into()));
        }
        let mut previous_end = None;
        let mut vintages = BTreeSet::new();
        for window in &windows {
            window
                .model
                .feature_schema()
                .validate()
                .map_err(|e| ModelError(e.to_string()))?;
            if window.model.feature_schema() != pipeline.schema() {
                return Err(ModelError("scheduled model feature schema mismatch".into()));
            }
            if previous_end.is_some_and(|end| end > window.valid_from_ns) {
                return Err(ModelError("model windows overlap or are unordered".into()));
            }
            if !vintages.insert(window.model.metadata().model_vintage.clone()) {
                return Err(ModelError("duplicate model vintage".into()));
            }
            previous_end = Some(window.valid_until_ns);
        }
        Ok(Self {
            pipeline,
            windows,
            mapper,
            warmup,
            cursor: 0,
            previous_time_ns: None,
        })
    }
}

impl<P: FeaturePipeline, M: Model, T: TargetMapper> Strategy for ScheduledMlStrategy<P, M, T> {
    type Error = ModelError;
    fn decide(&mut self, context: &DecisionContext<'_>) -> Result<TargetIntent, Self::Error> {
        let time = context.observation.time.timestamp_ns;
        if self
            .previous_time_ns
            .is_some_and(|previous| time < previous)
        {
            return Err(ModelError(
                "scheduled decisions cannot move backwards".into(),
            ));
        }
        while self.cursor < self.windows.len() && time >= self.windows[self.cursor].valid_until_ns {
            self.cursor += 1;
        }
        let window = self
            .windows
            .get_mut(self.cursor)
            .filter(|window| time >= window.valid_from_ns)
            .ok_or_else(|| {
                ModelError("no model available at decision time (future, gap, or expiry)".into())
            })?;
        self.previous_time_ns = Some(time);
        let output = self
            .pipeline
            .extract(context)
            .map_err(|e| ModelError(e.to_string()))?;
        let (mut intent, prediction) = match output {
            FeatureOutput::Warmup { .. } => (
                match self.warmup {
                    WarmupPolicy::Hold => context.hold(),
                    WarmupPolicy::Flat => context.target(0),
                }
                .state_code("ml_warmup"),
                None,
            ),
            FeatureOutput::Ready(features) => {
                features.validate().map_err(|e| ModelError(e.to_string()))?;
                if &features.schema != window.model.feature_schema() {
                    return Err(ModelError("pipeline changed feature schema".into()));
                }
                let prediction = window
                    .model
                    .predict(&features)
                    .map_err(|e| ModelError(e.to_string()))?;
                if !prediction.score.is_finite() {
                    return Err(ModelError("non-finite prediction".into()));
                }
                (
                    self.mapper
                        .map(context, &prediction)
                        .map_err(|e| ModelError(e.to_string()))?,
                    Some(prediction),
                )
            }
        };
        let vintage = &window.model.metadata().model_vintage;
        if intent
            .model_vintage
            .as_ref()
            .is_some_and(|value| value != vintage)
        {
            return Err(ModelError(
                "mapper cannot override the actual model vintage".into(),
            ));
        }
        intent.model_vintage = Some(vintage.clone());
        intent.diagnostic_json = json!({
            "ml": {"model_vintage": vintage, "model_type": window.model.metadata().model_type,
                "feature_schema_sha256": window.model.feature_schema().sha256,
                "training_report_sha256": window.training_report_sha256,
                "fit_asof_ns": window.fit_asof_ns, "valid_from_ns": window.valid_from_ns,
                "valid_until_ns": window.valid_until_ns, "prediction": prediction},
            "mapper": intent.diagnostic_json,
        });
        Ok(intent)
    }
}

use bullet_evaluation::{EventTime, MarketPoint};
use bullet_ml::*;
use bullet_strategy::{DecisionContext, Strategy, TargetIntent};
use std::convert::Infallible;

struct Features {
    schema: FeatureSchema,
}
impl FeaturePipeline for Features {
    type Error = FeatureError;
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }
    fn extract(&mut self, _: &DecisionContext<'_>) -> Result<FeatureOutput, Self::Error> {
        Ok(FeatureOutput::Ready(FeatureVector::new(
            self.schema.clone(),
            vec![1.0],
        )?))
    }
}
struct StatefulMapper {
    count: i64,
}
impl TargetMapper for StatefulMapper {
    type Error = Infallible;
    fn map(
        &mut self,
        _: &DecisionContext<'_>,
        _: &Prediction,
    ) -> Result<TargetIntent, Self::Error> {
        self.count += 1;
        Ok(TargetIntent::new(self.count))
    }
}
fn schema() -> FeatureSchema {
    FeatureSchema::new(vec!["x".into()]).unwrap()
}
fn window(id: &str, start: u64, end: u64) -> ModelWindow<LinearModel> {
    ModelWindow::new(
        start,
        start,
        end,
        LinearModel::new(ModelMetadata::new(id, "linear"), schema(), vec![1.0], 0.0).unwrap(),
    )
    .unwrap()
}
fn context(point: &MarketPoint) -> DecisionContext<'_> {
    DecisionContext {
        step: 0,
        instrument: "TEST",
        history: std::slice::from_ref(point),
        observation: point,
        execution_time: EventTime {
            timestamp_ns: point.time.timestamp_ns + 1,
            sequence: 0,
        },
        previous_target_units: 0,
    }
}
fn point(time: u64) -> MarketPoint {
    MarketPoint {
        time: EventTime {
            timestamp_ns: time,
            sequence: 0,
        },
        instrument: "TEST".into(),
        price: 100.0,
    }
}

#[test]
fn switches_on_decision_time_and_preserves_mapper_state() {
    let mut s = ScheduledMlStrategy::new(
        Features { schema: schema() },
        vec![window("v1", 10, 20), window("v2", 20, 30)],
        StatefulMapper { count: 0 },
        WarmupPolicy::Hold,
    )
    .unwrap();
    let first = s.decide(&context(&point(19))).unwrap();
    // Execution is at 20, but decision 19 must still use v1.
    assert_eq!(first.model_vintage.as_deref(), Some("v1"));
    assert_eq!(first.target_units, 1);
    let second = s.decide(&context(&point(20))).unwrap();
    assert_eq!(second.model_vintage.as_deref(), Some("v2"));
    assert_eq!(second.target_units, 2);
    assert!(s.decide(&context(&point(18))).is_err());
    assert!(s.decide(&context(&point(30))).is_err());
}
#[test]
fn rejects_future_gaps_overlap_duplicate_vintage_and_schema_mismatch() {
    let make = |windows| {
        ScheduledMlStrategy::new(
            Features { schema: schema() },
            windows,
            StatefulMapper { count: 0 },
            WarmupPolicy::Flat,
        )
    };
    assert!(make(vec![window("v1", 10, 21), window("v2", 20, 30)]).is_err());
    assert!(make(vec![window("v1", 10, 20), window("v1", 20, 30)]).is_err());
    let mut s = make(vec![window("v1", 10, 20), window("v2", 21, 30)]).unwrap();
    assert!(s.decide(&context(&point(9))).is_err());
    assert!(s.decide(&context(&point(20))).is_err());
    let model = LinearModel::new(
        ModelMetadata::new("other", "linear"),
        FeatureSchema::new(vec!["other".into()]).unwrap(),
        vec![1.0],
        0.0,
    )
    .unwrap();
    assert!(make(vec![ModelWindow::new(0, 0, 10, model).unwrap()]).is_err());
    let model = LinearModel::new(
        ModelMetadata::new("late", "linear"),
        schema(),
        vec![1.0],
        0.0,
    )
    .unwrap();
    assert!(ModelWindow::new(11, 10, 20, model).is_err());
}
#[test]
fn transform_roundtrip_validates_dimensions_and_nonfinite_values() {
    let transform = FeatureTransform::Standardize {
        mean: vec![2.0],
        scale: vec![2.0],
    };
    let json = serde_json::to_string(&transform).unwrap();
    let copy: FeatureTransform = serde_json::from_str(&json).unwrap();
    assert_eq!(
        copy.apply(&FeatureVector::new(schema(), vec![4.0]).unwrap())
            .unwrap()
            .values,
        vec![1.0]
    );
    let bad = FeatureTransform::Standardize {
        mean: vec![0.0],
        scale: vec![0.0],
    };
    assert!(
        bad.apply(&FeatureVector::new(schema(), vec![4.0]).unwrap())
            .is_err()
    );
}

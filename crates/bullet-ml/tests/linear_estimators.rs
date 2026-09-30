use bullet_ml::*;
fn linear(kind: &str, weights: Vec<f64>, bias: f64) -> LinearModel {
    LinearModel::new(
        ModelMetadata::new("test-v1", kind),
        FeatureSchema::new(vec!["x".into()]).unwrap(),
        weights,
        bias,
    )
    .unwrap()
}
#[test]
fn logistic_probabilities_remain_finite_at_extreme_margins_and_reload_exactly() {
    let mut model =
        LogisticRegressionModel::new(linear("linear-logistic-regression", vec![1.0], 0.0)).unwrap();
    let mut copy: LogisticRegressionModel =
        serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
    for margin in [-1000.0, -20.0, 0.0, 20.0, 1000.0] {
        let vector = FeatureVector::new(model.feature_schema().clone(), vec![margin]).unwrap();
        let p = model.predict(&vector).unwrap().score;
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(p, copy.predict(&vector).unwrap().score);
    }
    assert_eq!(logistic_probability(0.0), 0.5);
    assert_eq!(logistic_probability(1000.0), 1.0);
    assert_eq!(logistic_probability(-1000.0), 0.0);
}
#[test]
fn huber_scale_is_persisted_but_does_not_rescale_the_prediction() {
    let mut model =
        HuberRegressionModel::new(linear("huber-regression", vec![2.0], 1.0), 0.3).unwrap();
    let vector = FeatureVector::new(model.feature_schema().clone(), vec![3.0]).unwrap();
    assert_eq!(model.predict(&vector).unwrap().score, 7.0);
    let mut restored: HuberRegressionModel =
        serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
    assert_eq!(restored.scale(), 0.3);
    assert_eq!(restored.predict(&vector).unwrap().score, 7.0);
    assert!(HuberRegressionModel::new(linear("huber-regression", vec![2.0], 1.0), 0.0).is_err());
}
#[test]
fn serialized_class_order_schema_and_model_parameters_are_checked() {
    let model =
        LogisticRegressionModel::new(linear("linear-logistic-regression", vec![1.0], 0.0)).unwrap();
    let mut json = serde_json::to_value(&model).unwrap();
    json["classes"] = serde_json::json!([1, 0]);
    let bad: LogisticRegressionModel = serde_json::from_value(json).unwrap();
    assert!(bad.validate().is_err());
    let mut json = serde_json::to_value(&model).unwrap();
    json["linear"]["weights"] = serde_json::json!([]);
    let bad: LogisticRegressionModel = serde_json::from_value(json).unwrap();
    assert!(bad.validate().is_err());
    let mut model = model;
    assert!(
        model
            .predict(
                &FeatureVector::new(FeatureSchema::new(vec!["other".into()]).unwrap(), vec![1.0])
                    .unwrap()
            )
            .is_err()
    );
}

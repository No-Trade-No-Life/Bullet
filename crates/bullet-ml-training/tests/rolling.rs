use bullet_ml::{FeatureSchema, FeatureTransform, FeatureVector, LinearModel, ModelMetadata};
use bullet_ml_training::{
    ChronologicalSplit, TrainingDataset, TrainingError, TrainingExample, rolling::*,
};

fn dataset(mutate_future: bool) -> TrainingDataset {
    let schema = FeatureSchema::new(vec!["x".into()]).unwrap();
    let rows = (1..=15)
        .map(|i| {
            let x = if mutate_future && i >= 10 {
                10000.0
            } else {
                i as f64
            };
            let end = if i == 5 { 20 } else { i + 1 };
            TrainingExample::new(
                i,
                i,
                end,
                FeatureVector::new(schema.clone(), vec![x]).unwrap(),
                x * 2.0,
            )
            .unwrap()
        })
        .collect();
    TrainingDataset::new(schema, rows).unwrap()
}
struct Backend;
impl RollingTrainer for Backend {
    type Model = LinearModel;
    fn fit(&mut self, b: &FitBatch<'_>) -> Result<(Self::Model, FitReceipt), TrainingError> {
        b.validate()?;
        let model = LinearModel::new(
            ModelMetadata::new(b.model_vintage, "test-linear"),
            b.schema.clone(),
            vec![2.0],
            0.0,
        )
        .unwrap();
        let receipt = FitReceipt {
            backend: "test".into(),
            backend_version: "1".into(),
            parameters: serde_json::json!({}),
            model_sha256: model.artifact_sha256()?,
        };
        Ok((model, receipt))
    }
}
fn options() -> RollingOptions {
    RollingOptions {
        minimum_training_samples: 2,
        scaling: Scaling::Standardize,
        weight_normalization: bullet_ml_training::rolling::WeightNormalization::TrainingMeanOne,
        vintage_prefix: "test".into(),
    }
}
#[test]
fn sliding_window_maturity_and_gap_are_timestamp_based() {
    let data = dataset(false);
    let plan = RollingPlan::new(10, 16, 6, 3, 1, 0).unwrap();
    assert_eq!(
        plan.folds()[0].training_indices(&data).unwrap(),
        vec![3, 5, 6, 7]
    ); // decision 4,6,7,8
    assert_eq!(
        plan.folds()[1].training_indices(&data).unwrap(),
        vec![6, 7, 8, 9, 10]
    );
    assert_eq!(plan.folds()[1].train_start_ns, 7);
    let result = train_rolling(
        &data,
        &vec![1.0; data.len()],
        &plan,
        &options(),
        &mut Backend,
    )
    .unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].report.maximum_training_label_end_ns, 9);
    assert_eq!(result[0].report.model_vintage, "test@10");
    let FeatureTransform::Standardize { mean, .. } = &result[0].report.transform else {
        panic!("expected scaler")
    };
    assert_eq!(mean, &vec![6.25]);
    result[0].report.validate().unwrap();
}
#[test]
fn future_mutations_do_not_change_earlier_selection_scaler_or_report() {
    let a = dataset(false);
    let b = dataset(true);
    let p = RollingPlan::new(10, 13, 6, 3, 1, 0).unwrap();
    let a = train_rolling(&a, &vec![1.0; a.len()], &p, &options(), &mut Backend).unwrap();
    let b = train_rolling(&b, &vec![1.0; b.len()], &p, &options(), &mut Backend).unwrap();
    assert_eq!(a[0].report, b[0].report);
}
#[test]
fn revalidates_mutated_rows_and_checks_every_training_label() {
    let data = dataset(false);
    // Earlier row 5 ends at 20; last train row 8 ends at 9. Both must be checked.
    assert!(
        data.validate_split(ChronologicalSplit::new(data.len(), 8, 2).unwrap())
            .is_err()
    );
    let mut rows = data.examples().to_vec();
    rows[0].feature_end_ns = 100;
    assert!(TrainingDataset::new(data.schema().clone(), rows).is_err());
    let mut rows = data.examples().to_vec();
    rows[0].features.values[0] = f64::NAN;
    assert!(TrainingDataset::new(data.schema().clone(), rows).is_err());
}
#[test]
fn rejects_bad_windows_weights_insufficient_samples_and_tampered_report() {
    assert!(RollingPlan::new(10, 20, 0, 3, 0, 0).is_err());
    assert!(RollingPlan::new(10, 20, 6, 0, 0, 0).is_err());
    assert!(RollingPlan::new(10, 20, 11, 3, 0, 0).is_err());
    assert!(RollingPlan::new(10, 20, 6, 3, 0, 3).is_err());
    let data = dataset(false);
    let plan = RollingPlan::new(10, 13, 6, 3, 0, 0).unwrap();
    assert!(
        train_rolling(
            &data,
            &vec![0.0; data.len()],
            &plan,
            &options(),
            &mut Backend
        )
        .is_err()
    );
    let mut o = options();
    o.minimum_training_samples = 100;
    assert!(train_rolling(&data, &vec![1.0; data.len()], &plan, &o, &mut Backend).is_err());
    let mut folds = train_rolling(
        &data,
        &vec![1.0; data.len()],
        &plan,
        &options(),
        &mut Backend,
    )
    .unwrap();
    folds[0].report.fold.predict_start_ns = 9;
    assert!(folds.remove(0).into_window().is_err());
}
#[test]
fn activation_delay_is_explicit_and_prediction_tail_is_bounded() {
    let plan = RollingPlan::new(10, 18, 6, 3, 1, 1).unwrap();
    assert_eq!(
        (
            plan.folds()[0].fit_asof_ns,
            plan.folds()[0].predict_start_ns
        ),
        (10, 11)
    );
    assert_eq!(plan.folds().last().unwrap().predict_end_ns, 18);
    let json = serde_json::to_string(&plan).unwrap();
    let restored: RollingPlan = serde_json::from_str(&json).unwrap();
    restored.validate().unwrap();
    assert_eq!(plan.folds(), restored.folds());
}

#[test]
fn weights_are_normalized_on_selected_rows_only_and_wrong_models_cannot_attach_to_reports() {
    let data = dataset(false);
    let plan = RollingPlan::new(10, 13, 6, 3, 1, 0).unwrap();
    let mut weights: Vec<_> = (1..=data.len()).map(|i| i as f64).collect();
    let a = train_rolling(&data, &weights, &plan, &options(), &mut Backend).unwrap();
    for weight in &mut weights[9..] {
        *weight = 1e9;
    }
    let b = train_rolling(&data, &weights, &plan, &options(), &mut Backend).unwrap();
    assert_eq!(a[0].report, b[0].report);
    assert_ne!(
        a[0].report.raw_sample_weights_sha256,
        a[0].report.sample_weights_sha256
    );
    let mut fold = a.into_iter().next().unwrap();
    let wrong = LinearModel::new(
        ModelMetadata::new(&fold.report.model_vintage, "test-linear"),
        data.schema().clone(),
        vec![99.0],
        0.0,
    )
    .unwrap();
    fold.model = bullet_ml::PreprocessedModel::new(wrong, fold.report.transform.clone()).unwrap();
    assert!(fold.into_window().is_err());
}

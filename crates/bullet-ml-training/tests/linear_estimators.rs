#![cfg(feature = "linear-backend")]
use bullet_ml::{
    FeatureSchema, FeatureVector, HuberRegressionModel, LogisticRegressionModel, Model,
};
use bullet_ml_training::{TrainingDataset, TrainingExample, linear_estimators::*, rolling::*};
use serde_json::Value;

fn schema() -> FeatureSchema {
    FeatureSchema::new(vec!["a".into(), "b".into(), "c".into()]).unwrap()
}
fn assert_close(a: f64, b: f64, tol: f64, context: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{context}: {a} vs {b}, tolerance {tol}"
    );
}
#[test]
fn sklearn_151_weighted_objective_fixtures_match_each_coefficient_prediction_and_scale() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/linear-estimator-oracle.json")).unwrap();
    let coef_tol = fixture["tolerances"]["coefficient_atol"].as_f64().unwrap();
    let pred_tol = fixture["tolerances"]["prediction_atol"].as_f64().unwrap();
    let scale_tol = fixture["tolerances"]["scale_atol"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let schema = schema();
        let features: Vec<Vec<f64>> =
            serde_json::from_value(fixture["inputs"]["features"].clone()).unwrap();
        let targets: Vec<f64> = serde_json::from_value(
            fixture["inputs"][format!("{}_targets", case["family"].as_str().unwrap())].clone(),
        )
        .unwrap();
        let weights: Vec<f64> = serde_json::from_value(
            fixture["inputs"][if case["weighted"].as_bool().unwrap() {
                "weighted_weights"
            } else {
                "uniform_weights"
            }]
            .clone(),
        )
        .unwrap();
        let batch = FitBatch {
            schema: &schema,
            features: &features,
            targets: &targets,
            weights: &weights,
            model_vintage: case["name"].as_str().unwrap(),
        };
        let config = &case["config"];
        let expected = &case["expected"];
        let vectors: Vec<_> = features
            .iter()
            .map(|x| FeatureVector::new(schema.clone(), x.clone()).unwrap())
            .collect();
        let (coef, bias, predictions, scale, receipt) = if case["family"] == "logistic" {
            let mut trainer = LogisticRegressionTrainer::new(LogisticRegressionConfig {
                c: config["C"].as_f64().unwrap(),
                fit_intercept: config["fit_intercept"].as_bool().unwrap(),
                optimizer: LbfgsbConfig {
                    gradient_tolerance: 1e-9,
                    function_tolerance: 64.0 * f64::EPSILON,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
            let (mut model, receipt) = trainer.fit(&batch).unwrap();
            let coef = model.linear().weights.clone();
            let bias = model.linear().bias;
            let mut copy: LogisticRegressionModel =
                serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
            let predictions: Vec<_> = vectors
                .iter()
                .map(|x| model.predict(x).unwrap().score)
                .collect();
            assert_eq!(
                predictions,
                vectors
                    .iter()
                    .map(|x| copy.predict(x).unwrap().score)
                    .collect::<Vec<_>>()
            );
            assert_eq!(model.artifact_sha256().unwrap(), receipt.model_sha256);
            (coef, bias, predictions, None, receipt)
        } else {
            let mut trainer = HuberRegressionTrainer::new(HuberRegressionConfig {
                alpha: config["alpha"].as_f64().unwrap(),
                epsilon: config["epsilon"].as_f64().unwrap(),
                fit_intercept: config["fit_intercept"].as_bool().unwrap(),
                optimizer: LbfgsbConfig {
                    gradient_tolerance: 1e-9,
                    ..Default::default()
                },
            })
            .unwrap();
            let (mut model, receipt) = trainer.fit(&batch).unwrap();
            let coef = model.linear().weights.clone();
            let bias = model.linear().bias;
            let scale = model.scale();
            let mut copy: HuberRegressionModel =
                serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
            let predictions: Vec<_> = vectors
                .iter()
                .map(|x| model.predict(x).unwrap().score)
                .collect();
            assert_eq!(
                predictions,
                vectors
                    .iter()
                    .map(|x| copy.predict(x).unwrap().score)
                    .collect::<Vec<_>>()
            );
            assert_eq!(model.artifact_sha256().unwrap(), receipt.model_sha256);
            (coef, bias, predictions, Some(scale), receipt)
        };
        assert_eq!(
            coef.len(),
            expected["coefficients"].as_array().unwrap().len()
        );
        assert_eq!(
            predictions.len(),
            expected["predictions"].as_array().unwrap().len()
        );
        for (a, b) in coef
            .iter()
            .zip(expected["coefficients"].as_array().unwrap())
        {
            assert_close(*a, b.as_f64().unwrap(), coef_tol, batch.model_vintage);
        }
        assert_close(
            bias,
            expected["intercept"].as_f64().unwrap(),
            coef_tol,
            batch.model_vintage,
        );
        for (a, b) in predictions
            .iter()
            .zip(expected["predictions"].as_array().unwrap())
        {
            assert_close(*a, b.as_f64().unwrap(), pred_tol, batch.model_vintage);
        }
        if let Some(scale) = scale {
            assert_close(
                scale,
                expected["scale"].as_f64().unwrap(),
                scale_tol,
                batch.model_vintage,
            );
        }
        let max_error = |actual: &[f64], expected: &Value| {
            actual
                .iter()
                .zip(expected.as_array().unwrap())
                .map(|(a, b)| (a - b.as_f64().unwrap()).abs())
                .fold(0.0, f64::max)
        };
        eprintln!(
            "ORACLE_AGREEMENT {}",
            serde_json::json!({
                "case":batch.model_vintage, "coefficient_max_abs_error":max_error(&coef, &expected["coefficients"]),
                "intercept_abs_error":(bias-expected["intercept"].as_f64().unwrap()).abs(),
                "prediction_max_abs_error":max_error(&predictions, &expected["predictions"]),
                "scale_abs_error":scale.map(|v| (v-expected["scale"].as_f64().unwrap()).abs()),
            })
        );
        assert!(
            receipt.parameters["optimizer"]["iterations"]
                .as_u64()
                .unwrap()
                <= 1000
        );
        assert!(
            receipt.parameters["optimizer"]["evaluations"]
                .as_u64()
                .unwrap()
                <= 15000
        );
        assert!(
            receipt.parameters["optimizer"]["objective"]
                .as_f64()
                .unwrap()
                .is_finite()
        );
    }
}

fn dataset(family: &str, mutate: bool) -> TrainingDataset {
    let s = FeatureSchema::new(vec!["x".into()]).unwrap();
    let examples = (0..96)
        .map(|i| {
            let x = if mutate && i >= 48 {
                999.0
            } else {
                (i % 11) as f64
            };
            let y = if family == "logistic" {
                f64::from(i % 7 >= 3)
            } else {
                x * 1.7 + 0.3 + 0.1 * ((i * 7) % 9) as f64
            };
            TrainingExample::new(
                i,
                i,
                i + 2,
                FeatureVector::new(s.clone(), vec![x]).unwrap(),
                y,
            )
            .unwrap()
        })
        .collect();
    TrainingDataset::new(s, examples).unwrap()
}
fn options() -> RollingOptions {
    RollingOptions {
        minimum_training_samples: 16,
        scaling: Scaling::Standardize,
        weight_normalization: WeightNormalization::TrainingMeanOne,
        vintage_prefix: "rolling-linear".into(),
    }
}
fn prefix_check<B: RollingTrainer>(family: &str, backend: &mut B) {
    let a = dataset(family, false);
    let b = dataset(family, true);
    let plan = RollingPlan::new(48, 80, 32, 16, 1, 0).unwrap();
    let weights: Vec<_> = (0..96)
        .map(|i| if i % 3 == 0 { 2.0 } else { 1.0 })
        .collect();
    let mut future_weights = weights.clone();
    future_weights[48..].fill(100.0);
    let x = train_rolling(&a, &weights, &plan, &options(), backend).unwrap();
    let y = train_rolling(&b, &future_weights, &plan, &options(), backend).unwrap();
    assert_eq!(x[0].report, y[0].report);
    assert_ne!(
        x[1].report.training_dataset_sha256,
        y[1].report.training_dataset_sha256
    );
    assert_eq!(
        x[0].model.model().artifact_sha256().unwrap(),
        y[0].model.model().artifact_sha256().unwrap()
    );
    for fold in x {
        fold.into_window().unwrap();
    }
}
#[test]
fn both_backends_reuse_rolling_maturity_scaling_weights_and_prefix_invariance() {
    prefix_check(
        "logistic",
        &mut LogisticRegressionTrainer::new(Default::default()).unwrap(),
    );
    prefix_check(
        "huber",
        &mut HuberRegressionTrainer::new(Default::default()).unwrap(),
    );
}
#[test]
fn budgets_bad_labels_weights_and_invalid_configs_fail_closed() {
    let data = dataset("logistic", false);
    let features: Vec<_> = data
        .examples()
        .iter()
        .map(|r| r.features.values.clone())
        .collect();
    let targets: Vec<_> = data.examples().iter().map(|r| r.target).collect();
    let weights = vec![1.0; 96];
    let batch = FitBatch {
        schema: data.schema(),
        features: &features,
        targets: &targets,
        weights: &weights,
        model_vintage: "bad",
    };
    let mut logistic = LogisticRegressionTrainer::new(LogisticRegressionConfig {
        optimizer: LbfgsbConfig {
            max_evaluations: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    assert!(
        logistic
            .fit(&batch)
            .unwrap_err()
            .to_string()
            .contains("budget exhausted")
    );
    let mut logistic = LogisticRegressionTrainer::new(Default::default()).unwrap();
    assert!(
        logistic
            .fit(&FitBatch {
                targets: &[0.0; 96],
                ..batch
            })
            .is_err()
    );
    let invalid = vec![0.5; 96];
    assert!(
        logistic
            .fit(&FitBatch {
                targets: &invalid,
                ..batch
            })
            .is_err()
    );
    assert!(
        logistic
            .fit(&FitBatch {
                weights: &[0.0; 96],
                ..batch
            })
            .is_err()
    );
    assert!(
        LogisticRegressionTrainer::new(LogisticRegressionConfig {
            c: 0.0,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        HuberRegressionTrainer::new(HuberRegressionConfig {
            epsilon: 0.5,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        HuberRegressionTrainer::new(HuberRegressionConfig {
            alpha: -1.0,
            ..Default::default()
        })
        .is_err()
    );
    // A failed solve must not contaminate the next native solve's static work state.
    logistic.fit(&batch).unwrap();
}
#[test]
fn huber_handles_outliers_and_learns_a_positive_scale_including_the_active_floor() {
    let s = FeatureSchema::new(vec!["x".into()]).unwrap();
    let x: Vec<_> = (0..32).map(|i| vec![i as f64 / 10.0]).collect();
    let mut y: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, r)| 2.0 * r[0] + 1.0 + 0.02 * ((i % 3) as f64 - 1.0))
        .collect();
    y[10] += 100.0;
    let weights = vec![1.0; 32];
    let batch = FitBatch {
        schema: &s,
        features: &x,
        targets: &y,
        weights: &weights,
        model_vintage: "outlier",
    };
    let (model, receipt) = HuberRegressionTrainer::new(Default::default())
        .unwrap()
        .fit(&batch)
        .unwrap();
    assert!((model.linear().weights[0] - 2.0).abs() < 0.05);
    assert!((model.linear().bias - 1.0).abs() < 0.05);
    assert!(model.scale() > 0.0);
    assert!(receipt.parameters["outlier_count"].as_u64().unwrap() > 0);
    let zero = vec![0.0; 32];
    let (model, _) = HuberRegressionTrainer::new(Default::default())
        .unwrap()
        .fit(&FitBatch {
            targets: &zero,
            ..batch
        })
        .unwrap();
    assert!(model.scale() >= HUBER_SCALE_LOWER_BOUND);
    assert!(model.scale() < 1e-10);
}
#[test]
fn concurrent_fits_are_serialized_and_repeatable() {
    let run = || {
        let data = dataset("huber", false);
        let x: Vec<_> = data
            .examples()
            .iter()
            .map(|r| r.features.values.clone())
            .collect();
        let y: Vec<_> = data.examples().iter().map(|r| r.target).collect();
        let a = vec![1.0; 96];
        let (model, report) = HuberRegressionTrainer::new(Default::default())
            .unwrap()
            .fit(&FitBatch {
                schema: data.schema(),
                features: &x,
                targets: &y,
                weights: &a,
                model_vintage: "thread",
            })
            .unwrap();
        (
            model.artifact_sha256().unwrap(),
            serde_json::to_string(&report).unwrap(),
        )
    };
    let expected = run();
    let handles: Vec<_> = (0..4).map(|_| std::thread::spawn(run)).collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
}

#[test]
fn logistic_intercept_is_unpenalized_and_predicts_the_weighted_prior() {
    let schema = FeatureSchema::new(vec!["zero".into()]).unwrap();
    let features = vec![vec![0.0]; 10];
    let targets: Vec<_> = (0..10).map(|i| f64::from(i >= 2)).collect();
    let weights: Vec<_> = targets
        .iter()
        .map(|&y| if y == 1.0 { 2.0 } else { 1.0 })
        .collect();
    let batch = FitBatch {
        schema: &schema,
        features: &features,
        targets: &targets,
        weights: &weights,
        model_vintage: "intercept",
    };
    let mut trainer = LogisticRegressionTrainer::new(LogisticRegressionConfig {
        c: 1e-6,
        optimizer: LbfgsbConfig {
            gradient_tolerance: 1e-10,
            function_tolerance: 64.0 * f64::EPSILON,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let (mut model, _) = trainer.fit(&batch).unwrap();
    assert_close(
        model.linear().bias,
        8.0f64.ln(),
        1e-8,
        "unpenalized intercept",
    );
    assert_eq!(model.linear().weights, vec![0.0]);
    assert_close(
        model
            .predict(&FeatureVector::new(schema, vec![0.0]).unwrap())
            .unwrap()
            .score,
        16.0 / 18.0,
        1e-9,
        "weighted class prior",
    );
}

#[test]
fn multiplying_logistic_weights_is_equivalent_to_increasing_c() {
    let data = dataset("logistic", false);
    let features: Vec<_> = data
        .examples()
        .iter()
        .map(|r| r.features.values.clone())
        .collect();
    let targets: Vec<_> = data.examples().iter().map(|r| r.target).collect();
    let weights = vec![1.0; 96];
    let batch = FitBatch {
        schema: data.schema(),
        features: &features,
        targets: &targets,
        weights: &weights,
        model_vintage: "weights-c",
    };
    let fit = |c, batch: &FitBatch<'_>| {
        LogisticRegressionTrainer::new(LogisticRegressionConfig {
            c,
            optimizer: LbfgsbConfig {
                gradient_tolerance: 1e-10,
                function_tolerance: 64.0 * f64::EPSILON,
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap()
        .fit(batch)
        .unwrap()
    };
    let (a, ra) = fit(
        1.0,
        &FitBatch {
            weights: &[4.0; 96],
            ..batch
        },
    );
    let (b, rb) = fit(4.0, &batch);
    assert_eq!(a.artifact_sha256().unwrap(), b.artifact_sha256().unwrap());
    assert_eq!(
        ra.parameters["effective_l2_strength"],
        rb.parameters["effective_l2_strength"]
    );
    let (base, _) = fit(1.0, &batch);
    assert_ne!(
        base.artifact_sha256().unwrap(),
        a.artifact_sha256().unwrap()
    );
}

#[test]
fn numerical_overflow_and_underflow_reject_instead_of_emitting_models() {
    let schema = FeatureSchema::new(vec!["x".into()]).unwrap();
    let batch = FitBatch {
        schema: &schema,
        features: &[vec![1.0], vec![2.0]],
        targets: &[0.0, 1.0],
        weights: &[1.0, 1.0],
        model_vintage: "numerics",
    };
    let mut logistic = LogisticRegressionTrainer::new(Default::default()).unwrap();
    assert!(
        logistic
            .fit(&FitBatch {
                weights: &[f64::MAX, f64::MAX],
                ..batch
            })
            .is_err()
    );
    assert!(
        logistic
            .fit(&FitBatch {
                weights: &[f64::MIN_POSITIVE, f64::MAX],
                ..batch
            })
            .is_err()
    );
    let mut huber = HuberRegressionTrainer::new(Default::default()).unwrap();
    assert!(
        huber
            .fit(&FitBatch {
                targets: &[f64::MAX, -f64::MAX],
                ..batch
            })
            .is_err()
    );
    assert!(
        HuberRegressionTrainer::new(HuberRegressionConfig {
            epsilon: f64::MAX,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        LbfgsbConfig {
            history_size: 65,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LbfgsbConfig {
            max_iterations: 0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LbfgsbConfig {
            function_tolerance: f64::NAN,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    logistic.fit(&batch).unwrap();
}

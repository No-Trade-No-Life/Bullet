#![cfg(feature = "xgboost-backend")]
use bullet_ml::{FeatureSchema, FeatureVector, Model};
use bullet_ml_training::{TrainingDataset, TrainingExample, rolling::*, xgboost::*};
use xgb::{Booster, DMatrix, parameters::BoosterParameters};

#[test]
fn native_classification_regression_weights_and_artifact_roundtrip_match_wrapper() {
    for objective in [Objective::BinaryLogistic, Objective::SquaredError] {
        let schema = FeatureSchema::new(vec!["x".into(), "phase".into()]).unwrap();
        let features: Vec<_> = (0..64)
            .map(|i| vec![i as f64 / 64.0, (i % 7) as f64])
            .collect();
        let targets: Vec<_> = (0..64)
            .map(|i| {
                if objective == Objective::BinaryLogistic {
                    f64::from(i > 32)
                } else {
                    i as f64 / 32.0 + 0.1
                }
            })
            .collect();
        let weights: Vec<_> = (0..64)
            .map(|i| if i % 3 == 0 { 3.0 } else { 1.0 })
            .collect();
        let config = XgboostConfig {
            objective,
            rounds: 12,
            ..Default::default()
        };
        let mut trainer = XgboostTrainer::new(config.clone()).unwrap();
        let batch = FitBatch {
            schema: &schema,
            features: &features,
            targets: &targets,
            weights: &weights,
            model_vintage: "test-v1",
        };
        let (model, receipt) = trainer.fit(&batch).unwrap();
        let vectors: Vec<_> = features
            .iter()
            .map(|v| FeatureVector::new(schema.clone(), v.clone()).unwrap())
            .collect();
        let predictions = model.predict_rows(&vectors).unwrap();
        assert!(predictions.windows(2).any(|p| p[0] != p[1]));
        let mut data = DMatrix::from_dense(
            &features
                .iter()
                .flatten()
                .map(|v| *v as f32)
                .collect::<Vec<_>>(),
            64,
        )
        .unwrap();
        data.set_labels(&targets.iter().map(|v| *v as f32).collect::<Vec<_>>())
            .unwrap();
        data.set_weights(&weights.iter().map(|v| *v as f32).collect::<Vec<_>>())
            .unwrap();
        let mut reference =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&data]).unwrap();
        for pair in receipt.parameters["native_parameters"].as_array().unwrap() {
            reference
                .set_param(pair[0].as_str().unwrap(), pair[1].as_str().unwrap())
                .unwrap();
        }
        for round in 0..config.rounds {
            reference.update(&data, round as i32).unwrap();
        }
        assert_eq!(
            predictions,
            reference
                .predict(&data)
                .unwrap()
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        );
        let json = serde_json::to_string(model.artifact()).unwrap();
        let restored = XgboostModel::load(serde_json::from_str(&json).unwrap()).unwrap();
        assert_eq!(model.artifact().native_version(), [3, 2, 0]);
        assert_eq!(predictions, restored.predict_rows(&vectors).unwrap());
        let ones = vec![1.0; 64];
        let unweighted = FitBatch {
            weights: &ones,
            ..batch
        };
        let (unweighted, _) = trainer.fit(&unweighted).unwrap();
        assert_ne!(predictions, unweighted.predict_rows(&vectors).unwrap());
        let mut corrupted: serde_json::Value = serde_json::from_str(&json).unwrap();
        corrupted["model_json_sha256"] = serde_json::json!("0".repeat(64));
        assert!(XgboostModel::load(serde_json::from_value(corrupted).unwrap()).is_err());
    }
}

#[test]
fn native_rolling_prefix_is_invariant_under_future_mutation_and_artifacts_restore() {
    let schema = FeatureSchema::new(vec!["x".into()]).unwrap();
    let make = |mutate: bool| {
        TrainingDataset::new(
            schema.clone(),
            (0..80)
                .map(|i| {
                    let x = if mutate && i >= 40 {
                        999.0
                    } else {
                        (i % 11) as f64
                    };
                    TrainingExample::new(
                        i,
                        i,
                        i + 2,
                        FeatureVector::new(schema.clone(), vec![x]).unwrap(),
                        x * 2.0,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    };
    let plan = RollingPlan::new(40, 60, 32, 10, 1, 0).unwrap();
    let options = RollingOptions {
        minimum_training_samples: 16,
        scaling: Scaling::Standardize,
        weight_normalization: bullet_ml_training::rolling::WeightNormalization::TrainingMeanOne,
        vintage_prefix: "native".into(),
    };
    let mut trainer = XgboostTrainer::new(XgboostConfig {
        rounds: 8,
        ..Default::default()
    })
    .unwrap();
    let a = train_rolling(&make(false), &[1.0; 80], &plan, &options, &mut trainer).unwrap();
    let b = train_rolling(&make(true), &[1.0; 80], &plan, &options, &mut trainer).unwrap();
    assert_eq!(a[0].report, b[0].report);
    assert_eq!(
        a[0].model.model().artifact().sha256().unwrap(),
        b[0].model.model().artifact().sha256().unwrap()
    );
    assert_ne!(
        a[1].report.training_dataset_sha256,
        b[1].report.training_dataset_sha256
    );
    let report_json = serde_json::to_string(&a[0].report).unwrap();
    let report: FoldReport = serde_json::from_str(&report_json).unwrap();
    report.validate().unwrap();
    let artifact = a[0].model.model().artifact().clone();
    let mut restored =
        bullet_ml::PreprocessedModel::new(XgboostModel::load(artifact).unwrap(), report.transform)
            .unwrap();
    let vector = FeatureVector::new(schema, vec![3.0]).unwrap();
    let expected = restored.predict(&vector).unwrap().score;
    let mut original = a.into_iter().next().unwrap().model;
    assert_eq!(expected, original.predict(&vector).unwrap().score);
}

#[test]
fn native_backend_rejects_invalid_classes_weights_shapes_and_float_overflow() {
    let schema = FeatureSchema::new(vec!["x".into()]).unwrap();
    let features = vec![vec![1.0], vec![2.0]];
    let targets = vec![0.0, 2.0];
    let weights = vec![1.0, 1.0];
    let batch = FitBatch {
        schema: &schema,
        features: &features,
        targets: &targets,
        weights: &weights,
        model_vintage: "bad",
    };
    let mut trainer = XgboostTrainer::new(XgboostConfig {
        objective: Objective::BinaryLogistic,
        ..Default::default()
    })
    .unwrap();
    assert!(trainer.fit(&batch).is_err());
    let targets = vec![0.0, 0.0];
    assert!(
        trainer
            .fit(&FitBatch {
                targets: &targets,
                ..batch
            })
            .is_err()
    );
    let mut trainer = XgboostTrainer::new(XgboostConfig::default()).unwrap();
    let weights = vec![1.0, 0.0];
    assert!(
        trainer
            .fit(&FitBatch {
                weights: &weights,
                ..batch
            })
            .is_err()
    );
    let features = vec![vec![f64::MAX], vec![2.0]];
    assert!(
        trainer
            .fit(&FitBatch {
                features: &features,
                ..batch
            })
            .is_err()
    );
}

#[test]
fn continuous_oos_decision_prefix_survives_future_market_and_training_mutation() {
    use bullet_evaluation::{Accounting, EvaluationConfig, EventTime, MarketPoint, TerminalPolicy};
    use bullet_ml::{
        FeatureError, FeatureOutput, FeaturePipeline, ScheduledMlStrategy, ScoreToExposure,
        WarmupPolicy,
    };
    use bullet_strategy::{DecisionContext, StrategyConfig, StrategyMetadata, StrategyRunner};
    struct Pipeline(FeatureSchema);
    impl FeaturePipeline for Pipeline {
        type Error = FeatureError;
        fn schema(&self) -> &FeatureSchema {
            &self.0
        }
        fn extract(&mut self, c: &DecisionContext<'_>) -> Result<FeatureOutput, Self::Error> {
            Ok(FeatureOutput::Ready(FeatureVector::new(
                self.0.clone(),
                vec![c.observation.price],
            )?))
        }
    }
    let compile = |mutate: bool| {
        let schema = FeatureSchema::new(vec!["price".into()]).unwrap();
        let market: Vec<_> = (0..82)
            .map(|i| MarketPoint {
                time: EventTime {
                    timestamp_ns: i,
                    sequence: 0,
                },
                instrument: "TEST".into(),
                price: if mutate && i >= 50 {
                    1000.0
                } else {
                    100.0 + (i % 11) as f64
                },
            })
            .collect();
        let rows = (0..80)
            .map(|i| {
                TrainingExample::new(
                    i,
                    i,
                    i + 2,
                    FeatureVector::new(schema.clone(), vec![market[i as usize].price]).unwrap(),
                    market[i as usize].price * 2.0,
                )
                .unwrap()
            })
            .collect();
        let data = TrainingDataset::new(schema.clone(), rows).unwrap();
        let plan = RollingPlan::new(40, 80, 32, 10, 0, 0).unwrap();
        let options = RollingOptions {
            minimum_training_samples: 16,
            scaling: Scaling::None,
            weight_normalization: WeightNormalization::Preserve,
            vintage_prefix: "prefix".into(),
        };
        let mut backend = XgboostTrainer::new(XgboostConfig {
            rounds: 8,
            ..Default::default()
        })
        .unwrap();
        let folds = train_rolling(&data, &[1.0; 80], &plan, &options, &mut backend).unwrap();
        let windows = folds
            .into_iter()
            .map(|f| f.into_window().unwrap())
            .collect();
        let strategy = ScheduledMlStrategy::new(
            Pipeline(schema),
            windows,
            ScoreToExposure::new(211.0, 205.0, 1, -1).unwrap(),
            WarmupPolicy::Hold,
        )
        .unwrap();
        let config = StrategyConfig {
            metadata: StrategyMetadata::default(),
            evaluation: EvaluationConfig {
                accounting: Accounting::NormalizedExposureV1,
                instrument: "TEST".into(),
                one_way_cost_bps: 4.0,
                slippage_bps: 0.0,
                terminal_policy: TerminalPolicy::Liquidate,
                sharpe_periods_per_year: 365.0,
                sharpe_standard_deviation_ddof: 1,
                annualization_days_per_year: 365.2425,
                evaluation_days: Vec::new(),
            },
        };
        StrategyRunner::new(strategy, market[40..].to_vec(), config)
            .unwrap()
            .compile()
            .unwrap()
    };
    let a = compile(false);
    let b = compile(true);
    assert_eq!(
        serde_json::to_vec(&a.decisions[..10]).unwrap(),
        serde_json::to_vec(&b.decisions[..10]).unwrap()
    );
    assert_eq!(a.decisions[0].model_vintage.as_deref(), Some("prefix@40"));
    assert_eq!(a.decisions[10].model_vintage.as_deref(), Some("prefix@50"));
    assert_ne!(
        serde_json::to_vec(&a.decisions[20..]).unwrap(),
        serde_json::to_vec(&b.decisions[20..]).unwrap()
    );
}

#[test]
fn estimated_intercept_matches_independent_sklearn_weighted_and_unweighted_oracles() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/xgboost-intercept-oracle.json")).unwrap();
    let schema = FeatureSchema::new(vec!["a".into(), "b".into(), "c".into()]).unwrap();
    let features: Vec<Vec<f64>> = serde_json::from_value(oracle["features"].clone()).unwrap();
    let prediction_features: Vec<Vec<f64>> =
        serde_json::from_value(oracle["prediction_features"].clone()).unwrap();
    let rows: Vec<_> = prediction_features
        .into_iter()
        .map(|x| FeatureVector::new(schema.clone(), x).unwrap())
        .collect();
    for case in oracle["cases"].as_array().unwrap() {
        let objective = if case["kind"] == "classification" {
            Objective::BinaryLogistic
        } else {
            Objective::SquaredError
        };
        let targets: Vec<f64> = serde_json::from_value(case["targets"].clone()).unwrap();
        let weights: Vec<f64> = serde_json::from_value(case["weights"].clone()).unwrap();
        let expected: Vec<f64> = serde_json::from_value(case["predictions"].clone()).unwrap();
        let (model, receipt) = XgboostTrainer::new(XgboostConfig {
            objective,
            rounds: 128,
            learning_rate: 0.05,
            min_child_weight: 20.0,
            reg_lambda: 5.0,
            seed: 20260903,
            ..Default::default()
        })
        .unwrap()
        .fit(&FitBatch {
            schema: &schema,
            features: &features,
            targets: &targets,
            weights: &weights,
            model_vintage: "sklearn-intercept",
        })
        .unwrap();
        assert_eq!(
            model.predict_rows(&rows).unwrap(),
            expected,
            "{} weighted={}",
            case["kind"],
            case["weighted"]
        );
        let artifact = serde_json::to_value(model.artifact()).unwrap();
        let native: serde_json::Value =
            serde_json::from_str(artifact["model_json"].as_str().unwrap()).unwrap();
        assert_eq!(
            native["learner"]["learner_model_param"],
            case["initial_intercept"]
        );
        assert_eq!(
            native["learner"]["learner_model_param"]["boost_from_average"],
            "1"
        );
        assert_ne!(
            native["learner"]["learner_model_param"]["base_score"],
            "[5E-1]"
        );
        assert_eq!(
            receipt.parameters["initial_intercept"],
            "estimated_from_training_data"
        );
    }
}

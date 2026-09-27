use bullet_ml::{FeatureSchema, FeatureVector};
use bullet_ml_training::{
    ChronologicalSplit, LinearSolver, LinearTrainingConfig, TrainingDataset, TrainingExample,
    train_linear,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = FeatureSchema::new(vec!["return_1".into()])?;
    let examples = (0..12)
        .map(|index| {
            let feature = index as f64 / 100.0;
            let values = FeatureVector::new(schema.clone(), vec![feature])
                .map_err(|error| bullet_ml_training::TrainingError(error.to_string()))?;
            TrainingExample::new(index, index, index + 1, values, 2.0 * feature + 0.01)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dataset = TrainingDataset::new(schema, examples)?;
    let trained = train_linear(
        &dataset,
        ChronologicalSplit::new(dataset.len(), 8, 4)?,
        &LinearTrainingConfig {
            model_vintage: "linear-demo-v1".into(),
            model_type: "ordinary-least-squares".into(),
            solver: LinearSolver::OrdinaryLeastSquares,
        },
    )?;
    println!("dataset_sha256={}", trained.report.dataset_sha256);
    println!("validation_mse={}", trained.report.validation_mse);
    println!("model={}", serde_json::to_string_pretty(&trained.model)?);
    Ok(())
}

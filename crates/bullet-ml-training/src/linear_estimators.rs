//! Weighted linear logistic and concomitant-scale Huber fits using native L-BFGS-B.
pub use crate::lbfgsb_solver::{LbfgsbConfig, SOLVER_VERSION, SolverReport};
use crate::{
    TrainingError,
    lbfgsb_solver::minimize,
    rolling::{ArtifactModel, FitBatch, FitReceipt, RollingTrainer},
};
use bullet_ml::{
    HuberRegressionModel, LinearModel, LogisticRegressionModel, ModelMetadata, logistic_probability,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub const HUBER_SCALE_LOWER_BOUND: f64 = 10.0 * f64::EPSILON;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogisticRegressionConfig {
    /// Inverse L2 regularization strength. The intercept is never penalized.
    pub c: f64,
    pub fit_intercept: bool,
    pub minimum_rows_per_class: usize,
    pub optimizer: LbfgsbConfig,
}
impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            fit_intercept: true,
            minimum_rows_per_class: 1,
            optimizer: LbfgsbConfig {
                gradient_tolerance: 1e-4,
                function_tolerance: 64.0 * f64::EPSILON,
                ..Default::default()
            },
        }
    }
}
impl LogisticRegressionConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        self.optimizer.validate()?;
        if !self.c.is_finite()
            || self.c <= 0.0
            || !self.c.recip().is_finite()
            || self.minimum_rows_per_class == 0
        {
            return Err(TrainingError("logistic C must be positive finite with finite inverse; class minimum must be positive".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HuberRegressionConfig {
    pub epsilon: f64,
    pub alpha: f64,
    pub fit_intercept: bool,
    pub optimizer: LbfgsbConfig,
}
impl Default for HuberRegressionConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.35,
            alpha: 0.0001,
            fit_intercept: true,
            optimizer: LbfgsbConfig::default(),
        }
    }
}
impl HuberRegressionConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        self.optimizer.validate()?;
        if !self.epsilon.is_finite()
            || self.epsilon < 1.0
            || !(self.epsilon * self.epsilon).is_finite()
            || !self.alpha.is_finite()
            || self.alpha < 0.0
        {
            return Err(TrainingError(
                "Huber requires finite epsilon >= 1 and alpha >= 0".into(),
            ));
        }
        Ok(())
    }
}

pub struct LogisticRegressionTrainer {
    config: LogisticRegressionConfig,
}
impl LogisticRegressionTrainer {
    pub fn new(config: LogisticRegressionConfig) -> Result<Self, TrainingError> {
        config.validate()?;
        Ok(Self { config })
    }
}
pub struct HuberRegressionTrainer {
    config: HuberRegressionConfig,
}
impl HuberRegressionTrainer {
    pub fn new(config: HuberRegressionConfig) -> Result<Self, TrainingError> {
        config.validate()?;
        Ok(Self { config })
    }
}

impl RollingTrainer for LogisticRegressionTrainer {
    type Model = LogisticRegressionModel;
    fn fit(&mut self, batch: &FitBatch<'_>) -> Result<(Self::Model, FitReceipt), TrainingError> {
        batch.validate()?;
        let mut counts = [0usize; 2];
        for &y in batch.targets {
            if y != 0.0 && y != 1.0 {
                return Err(TrainingError(
                    "linear logistic labels must be exactly 0 and 1".into(),
                ));
            }
            counts[y as usize] += 1;
        }
        if counts
            .iter()
            .any(|&n| n < self.config.minimum_rows_per_class)
        {
            return Err(TrainingError("insufficient logistic class coverage".into()));
        }
        let sum = weight_sum(batch)?;
        let lambda = self.config.c.recip() / sum;
        if !lambda.is_finite() || lambda <= 0.0 || batch.weights.iter().any(|w| w / sum <= 0.0) {
            return Err(TrainingError(
                "logistic normalization/regularization overflows or underflows".into(),
            ));
        }
        let p = batch.schema.names.len();
        let dim = p + usize::from(self.config.fit_intercept);
        let (parameters, report) = minimize(
            vec![0.0; dim],
            vec![None; dim],
            &self.config.optimizer,
            |x, g| logistic_loss_gradient(batch, x, g, self.config.fit_intercept, lambda, sum),
        )?;
        let model = LogisticRegressionModel::new(
            LinearModel::new(
                ModelMetadata::new(batch.model_vintage, "linear-logistic-regression"),
                batch.schema.clone(),
                parameters[..p].to_vec(),
                if self.config.fit_intercept {
                    parameters[p]
                } else {
                    0.0
                },
            )
            .map_err(error)?,
        )
        .map_err(error)?;
        let receipt = FitReceipt {
            backend: "linear-logistic-lbfgsb".into(),
            backend_version: SOLVER_VERSION.into(),
            parameters: json!({"config":self.config,"optimizer":report,"objective_contract":"weighted-mean-binary-logloss+l2/(2*C*sum_weights)","effective_l2_strength":lambda,
                "weight_sum":sum,"class_counts":counts,"classes":[0,1],"intercept_penalized":false,"dtype":"float64","initialization":"zeros","native_line_search_limit":20,"native_calls":"serialized"}),
            model_sha256: model.artifact_sha256()?,
        };
        Ok((model, receipt))
    }
}
impl RollingTrainer for HuberRegressionTrainer {
    type Model = HuberRegressionModel;
    fn fit(&mut self, batch: &FitBatch<'_>) -> Result<(Self::Model, FitReceipt), TrainingError> {
        batch.validate()?;
        let sum = weight_sum(batch)?;
        let p = batch.schema.names.len();
        let dim = p + usize::from(self.config.fit_intercept) + 1;
        let mut initial = vec![0.0; dim];
        initial[dim - 1] = 1.0;
        let mut bounds = vec![None; dim];
        bounds[dim - 1] = Some(HUBER_SCALE_LOWER_BOUND);
        let (parameters, report) = minimize(initial, bounds, &self.config.optimizer, |x, g| {
            huber_loss_gradient(batch, x, g, &self.config)
        })?;
        let model = HuberRegressionModel::new(
            LinearModel::new(
                ModelMetadata::new(batch.model_vintage, "huber-regression"),
                batch.schema.clone(),
                parameters[..p].to_vec(),
                if self.config.fit_intercept {
                    parameters[p]
                } else {
                    0.0
                },
            )
            .map_err(error)?,
            parameters[dim - 1],
        )
        .map_err(error)?;
        let outlier_count = batch
            .features
            .iter()
            .zip(batch.targets)
            .filter(|(row, y)| {
                (**y - dot(row, &parameters[..p]) - model.linear().bias).abs()
                    > self.config.epsilon * model.scale()
            })
            .count();
        let receipt = FitReceipt {
            backend: "huber-lbfgsb".into(),
            backend_version: SOLVER_VERSION.into(),
            parameters: json!({"config":self.config,"optimizer":report,"objective_contract":"sum(weight*(sigma+sigma*huber(residual/sigma)))+alpha*l2",
                "weight_sum":sum,"fitted_scale":model.scale(),"scale_lower_bound":HUBER_SCALE_LOWER_BOUND,"outlier_count":outlier_count,
                "intercept_penalized":false,"dtype":"float64","initialization":"zero_coefficients_and_intercept,scale=1","native_line_search_limit":20,"native_calls":"serialized"}),
            model_sha256: model.artifact_sha256()?,
        };
        Ok((model, receipt))
    }
}

fn weight_sum(batch: &FitBatch<'_>) -> Result<f64, TrainingError> {
    let sum = batch.weights.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(TrainingError("invalid total sample weight".into()));
    }
    Ok(sum)
}
fn dot(x: &[f64], w: &[f64]) -> f64 {
    x.iter().zip(w).map(|(x, w)| x * w).sum()
}
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
fn logistic_loss_gradient(
    batch: &FitBatch<'_>,
    parameters: &[f64],
    gradient: &mut [f64],
    intercept: bool,
    lambda: f64,
    weight_sum: f64,
) -> Result<f64, TrainingError> {
    let p = batch.schema.names.len();
    let b = if intercept { parameters[p] } else { 0.0 };
    let mut loss = 0.0;
    gradient.fill(0.0);
    for ((row, &y), &weight) in batch.features.iter().zip(batch.targets).zip(batch.weights) {
        let z = dot(row, &parameters[..p]) + b;
        if !z.is_finite() {
            return Err(TrainingError("non-finite logistic margin".into()));
        }
        let a = weight / weight_sum;
        loss += a * softplus(if y == 1.0 { -z } else { z });
        let residual = if y == 1.0 {
            -logistic_probability(-z)
        } else {
            logistic_probability(z)
        };
        for (g, x) in gradient[..p].iter_mut().zip(row) {
            *g += a * residual * x;
        }
        if intercept {
            gradient[p] += a * residual;
        }
    }
    for (g, &w) in gradient[..p].iter_mut().zip(&parameters[..p]) {
        loss += 0.5 * lambda * w * w;
        *g += lambda * w;
    }
    Ok(loss)
}
fn huber_loss_gradient(
    batch: &FitBatch<'_>,
    parameters: &[f64],
    gradient: &mut [f64],
    config: &HuberRegressionConfig,
) -> Result<f64, TrainingError> {
    let p = batch.schema.names.len();
    let last = parameters.len() - 1;
    let sigma = parameters[last];
    // Native line-search arithmetic can put a positive trial slightly below the
    // bound. The objective is defined for all sigma > 0; minimize() separately
    // enforces the exact lower bound on the returned solution. Do not clip x.
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(TrainingError(format!(
            "invalid Huber trial scale: {sigma:e}"
        )));
    }
    let b = if config.fit_intercept {
        parameters[p]
    } else {
        0.0
    };
    let threshold = config.epsilon * sigma;
    if !threshold.is_finite() {
        return Err(TrainingError("Huber threshold overflow".into()));
    }
    gradient.fill(0.0);
    let mut loss = 0.0;
    for ((row, &y), &a) in batch.features.iter().zip(batch.targets).zip(batch.weights) {
        let r = y - dot(row, &parameters[..p]) - b;
        let (row_loss, slope, scale_gradient) = if r.abs() > threshold {
            (
                sigma * (1.0 - config.epsilon * config.epsilon) + 2.0 * config.epsilon * r.abs(),
                -2.0 * config.epsilon * r.signum(),
                1.0 - config.epsilon * config.epsilon,
            )
        } else {
            let q = r / sigma;
            (sigma + r * q, -2.0 * q, 1.0 - q * q)
        };
        loss += a * row_loss;
        for (g, x) in gradient[..p].iter_mut().zip(row) {
            *g += a * slope * x;
        }
        if config.fit_intercept {
            gradient[p] += a * slope;
        }
        gradient[last] += a * scale_gradient;
    }
    for (g, &w) in gradient[..p].iter_mut().zip(&parameters[..p]) {
        loss += config.alpha * w * w;
        *g += 2.0 * config.alpha * w;
    }
    Ok(loss)
}
fn error(e: impl std::fmt::Display) -> TrainingError {
    TrainingError(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bullet_ml::FeatureSchema;
    fn check_gradient(x: Vec<f64>, f: impl Fn(&[f64], &mut [f64]) -> Result<f64, TrainingError>) {
        let mut gradient = vec![0.0; x.len()];
        f(&x, &mut gradient).unwrap();
        for i in 0..x.len() {
            let mut left = x.clone();
            let mut right = x.clone();
            left[i] -= 1e-6;
            right[i] += 1e-6;
            let mut g = vec![0.0; x.len()];
            let numeric = (f(&right, &mut g).unwrap() - f(&left, &mut g).unwrap()) / 2e-6;
            assert!(
                (numeric - gradient[i]).abs() < 1e-7,
                "coordinate {i}: {numeric} vs {}",
                gradient[i]
            );
        }
    }

    #[test]
    fn analytic_gradients_match_central_differences_including_huber_scale() {
        let schema = FeatureSchema::new(vec!["a".into(), "b".into()]).unwrap();
        let features = vec![
            vec![-1.0, 0.5],
            vec![0.3, -0.7],
            vec![2.0, 1.0],
            vec![4.0, -2.0],
        ];
        let targets = vec![0.0, 1.0, 0.0, 1.0];
        let weights = vec![0.5, 2.0, 1.0, 3.0];
        let batch = FitBatch {
            schema: &schema,
            features: &features,
            targets: &targets,
            weights: &weights,
            model_vintage: "gradient",
        };
        check_gradient(vec![0.2, -0.1, 0.3], |x, g| {
            logistic_loss_gradient(&batch, x, g, true, 0.2, 6.5)
        });
        check_gradient(vec![0.2, -0.1, 0.3, 0.4], |x, g| {
            huber_loss_gradient(
                &batch,
                x,
                g,
                &HuberRegressionConfig {
                    alpha: 0.2,
                    ..Default::default()
                },
            )
        });
    }
    #[test]
    fn positive_huber_trial_below_bound_has_an_objective_without_clipping() {
        let schema = FeatureSchema::new(vec!["x".into()]).unwrap();
        let batch = FitBatch {
            schema: &schema,
            features: &[vec![0.0]],
            targets: &[0.0],
            weights: &[1.0],
            model_vintage: "domain",
        };
        // Reproduces native line-search cancellation observed in rolling_linear.
        let sigma = 2.2040839260883498e-15;
        assert!(sigma < HUBER_SCALE_LOWER_BOUND);
        let mut g = [0.0; 3];
        let config = HuberRegressionConfig::default();
        let loss = huber_loss_gradient(&batch, &[0.0, 0.0, sigma], &mut g, &config).unwrap();
        assert_eq!(loss, sigma);
        assert_eq!(g, [0.0, 0.0, 1.0]);
        for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(huber_loss_gradient(&batch, &[0.0, 0.0, sigma], &mut g, &config).is_err());
        }
    }
}

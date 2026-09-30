//! Bounded reverse-communication adapter for the pinned community native solver.
use crate::TrainingError;
use serde::{Deserialize, Serialize};
use std::{ffi::c_long, sync::Mutex};

pub const SOLVER_VERSION: &str = "lbfgsb-0.1.1/L-BFGS-B-C-3.0";
// The upstream C translation contains static work variables. Keep the entire
// solve, not just individual calls, serialized across Bullet trainers.
static SOLVER_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LbfgsbConfig {
    pub history_size: usize,
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub gradient_tolerance: f64,
    pub function_tolerance: f64,
}
impl Default for LbfgsbConfig {
    fn default() -> Self {
        Self {
            history_size: 10,
            max_iterations: 1000,
            max_evaluations: 15000,
            gradient_tolerance: 1e-5,
            function_tolerance: 1e7 * f64::EPSILON,
        }
    }
}
impl LbfgsbConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.history_size == 0
            || self.history_size > 64
            || self.max_iterations == 0
            || self.max_evaluations == 0
            || !self.gradient_tolerance.is_finite()
            || self.gradient_tolerance <= 0.0
            || !self.function_tolerance.is_finite()
            || self.function_tolerance < 0.0
            || self.function_tolerance >= 1.0
        {
            return Err(TrainingError(
                "invalid L-BFGS-B memory/budget/tolerances".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SolverReport {
    pub solver_version: String,
    pub iterations: usize,
    pub evaluations: usize,
    pub objective: f64,
    pub projected_gradient_infinity_norm: f64,
    pub termination: String,
}

pub(crate) fn minimize(
    mut x: Vec<f64>,
    lower: Vec<Option<f64>>,
    config: &LbfgsbConfig,
    mut evaluate: impl FnMut(&[f64], &mut [f64]) -> Result<f64, TrainingError>,
) -> Result<(Vec<f64>, SolverReport), TrainingError> {
    config.validate()?;
    if x.is_empty()
        || lower.len() != x.len()
        || x.iter().any(|v| !v.is_finite())
        || lower
            .iter()
            .zip(&x)
            .any(|(bound, v)| bound.is_some_and(|b| !b.is_finite() || *v < b))
    {
        return Err(TrainingError(
            "invalid solver initial point or lower bounds".into(),
        ));
    }
    let n =
        c_long::try_from(x.len()).map_err(|_| TrainingError("solver dimension overflow".into()))?;
    let m = c_long::try_from(config.history_size)
        .map_err(|_| TrainingError("solver memory overflow".into()))?;
    let size = (2 * config.history_size + 5)
        .checked_mul(x.len())
        .and_then(|v| {
            v.checked_add(11 * config.history_size * config.history_size + 8 * config.history_size)
        })
        .ok_or_else(|| TrainingError("solver workspace overflow".into()))?;
    let mut wa = vec![0.0; size];
    let mut iwa = vec![
        0 as c_long;
        x.len()
            .checked_mul(3)
            .ok_or_else(|| TrainingError("solver workspace overflow".into()))?
    ];
    let l: Vec<_> = lower.iter().map(|b| b.unwrap_or(0.0)).collect();
    let u = vec![0.0; x.len()];
    let nbd: Vec<c_long> = lower
        .iter()
        .map(|b| if b.is_some() { 1 } else { 0 })
        .collect();
    let mut g = vec![0.0; x.len()];
    let mut f = 0.0;
    let mut task: c_long = 1; // START in the versioned native header.
    let mut csave = [0 as c_long; 60];
    let mut lsave = [0 as c_long; 4];
    let mut isave = [0 as c_long; 44];
    let mut dsave = [0.0; 29];
    let factr = config.function_tolerance / f64::EPSILON;
    let mut iterations = 0;
    let mut evaluations = 0;
    let _guard = SOLVER_LOCK
        .lock()
        .map_err(|_| TrainingError("native optimizer lock poisoned".into()))?;
    loop {
        // SAFETY: LP64 target scalars match the crate's C `long` ABI. Every array
        // has the documented n/m-dependent size and distinct live backing storage.
        // C retains state only in these buffers/static work variables between calls;
        // the lock spans the full solve. No pointer escapes this private adapter.
        unsafe {
            lbfgsb::setulb(
                &n,
                &m,
                x.as_mut_ptr(),
                l.as_ptr(),
                u.as_ptr(),
                nbd.as_ptr(),
                &mut f,
                g.as_mut_ptr(),
                &factr,
                &config.gradient_tolerance,
                wa.as_mut_ptr(),
                iwa.as_mut_ptr(),
                &mut task,
                &(-1 as c_long),
                csave.as_mut_ptr(),
                lsave.as_mut_ptr(),
                isave.as_mut_ptr(),
                dsave.as_mut_ptr(),
            );
        }
        match task {
            10..=15 => {
                if iterations >= config.max_iterations || evaluations >= config.max_evaluations {
                    return Err(TrainingError(format!(
                        "L-BFGS-B budget exhausted: iterations={iterations}, evaluations={evaluations}"
                    )));
                }
                if x.iter().any(|v| !v.is_finite()) {
                    return Err(TrainingError("non-finite native trial point".into()));
                }
                g.fill(0.0);
                f = evaluate(&x, &mut g)?;
                evaluations += 1;
                if !f.is_finite() || g.iter().any(|v| !v.is_finite()) {
                    return Err(TrainingError("non-finite objective or gradient".into()));
                }
            }
            2 => {
                iterations += 1;
                if iterations > config.max_iterations {
                    return Err(TrainingError("L-BFGS-B iteration budget exhausted".into()));
                }
            }
            21 | 22 => {
                if iterations > config.max_iterations
                    || evaluations == 0
                    || !f.is_finite()
                    || !dsave[12].is_finite()
                    || x.iter().any(|v| !v.is_finite())
                    || lower.iter().zip(&x).any(|(b, v)| b.is_some_and(|b| *v < b))
                {
                    return Err(TrainingError("invalid native optimizer solution".into()));
                }
                return Ok((
                    x,
                    SolverReport {
                        solver_version: SOLVER_VERSION.into(),
                        iterations,
                        evaluations,
                        objective: f,
                        projected_gradient_infinity_norm: dsave[12],
                        termination: if task == 21 {
                            "projected_gradient"
                        } else {
                            "relative_function_reduction"
                        }
                        .into(),
                    },
                ));
            }
            _ => {
                return Err(TrainingError(format!(
                    "L-BFGS-B failed: native task={task}, iterations={iterations}, evaluations={evaluations}"
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solves_an_active_bound_and_unconstrained_coordinate() {
        let (x, report) = minimize(
            vec![1.0, 0.0],
            vec![Some(0.1), None],
            &LbfgsbConfig::default(),
            |x, g| {
                g[0] = 2.0 * x[0];
                g[1] = 2.0 * (x[1] - 3.0);
                Ok(x[0] * x[0] + (x[1] - 3.0).powi(2))
            },
        )
        .unwrap();
        assert!((x[0] - 0.1).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-8);
        assert!(report.projected_gradient_infinity_norm <= 1e-5);
    }
    #[test]
    fn failures_and_exhaustion_are_not_successful_models() {
        let config = LbfgsbConfig {
            max_evaluations: 1,
            ..Default::default()
        };
        assert!(
            minimize(vec![10.0], vec![None], &config, |x, g| {
                g[0] = 2.0 * x[0];
                Ok(x[0] * x[0])
            })
            .unwrap_err()
            .to_string()
            .contains("budget exhausted")
        );
        assert!(
            minimize(vec![0.0], vec![None], &LbfgsbConfig::default(), |_, _| Ok(
                f64::INFINITY
            ))
            .is_err()
        );
        assert!(
            minimize(
                vec![0.0],
                vec![Some(1.0)],
                &LbfgsbConfig::default(),
                |_, _| Ok(0.0)
            )
            .is_err()
        );
    }
    #[test]
    fn iteration_limit_nonfinite_gradient_and_abnormal_line_search_fail_closed() {
        let config = LbfgsbConfig {
            max_iterations: 1,
            ..Default::default()
        };
        let err = minimize(vec![10.0], vec![None], &config, |x, g| {
            g[0] = 2.0 * x[0];
            Ok(x[0] * x[0])
        })
        .unwrap_err();
        assert!(err.to_string().contains("budget exhausted"));
        let err = minimize(vec![0.0], vec![None], &LbfgsbConfig::default(), |_, g| {
            g[0] = f64::NAN;
            Ok(0.0)
        })
        .unwrap_err();
        assert!(err.to_string().contains("non-finite objective or gradient"));
        // Deliberately inconsistent objective/gradient cannot satisfy the line search.
        let err = minimize(vec![0.0], vec![None], &LbfgsbConfig::default(), |_, g| {
            g[0] = 1.0;
            Ok(0.0)
        })
        .unwrap_err();
        assert!(err.to_string().contains("native task=3"), "{err}");
    }
}

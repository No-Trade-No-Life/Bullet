#[derive(Clone, Debug, PartialEq)]
pub struct FeatureVector {
    pub values: Vec<f64>,
}

impl FeatureVector {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }
}

use bullet::{BarContext, Order, Strategy};

pub struct DualMovingAverage { closes: Vec<f64>, fast: usize, slow: usize }

pub fn strategy() -> DualMovingAverage { DualMovingAverage { closes: Vec::new(), fast: 20, slow: 50 } }

impl Strategy for DualMovingAverage {
    fn on_bar(&mut self, context: BarContext<'_>) -> Order {
        if context.instrument != "IM8888" { return Order::None; }
        self.closes.push(context.bar.close.value());
        if self.closes.len() < self.slow { return Order::None; }
        let fast = average(&self.closes[self.closes.len() - self.fast..]);
        let slow = average(&self.closes[self.closes.len() - self.slow..]);
        if fast > slow && context.position == 0 { Order::Buy(1) } else if fast <= slow && context.position > 0 { Order::Close } else { Order::None }
    }
}
fn average(values: &[f64]) -> f64 { values.iter().sum::<f64>() / values.len() as f64 }

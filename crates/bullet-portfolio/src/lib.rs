use bullet_execution::{Fill, Side};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Position {
    pub quantity: i64,
}

impl Position {
    pub fn apply(&mut self, side: Side, fill: Fill) {
        let delta = fill.quantity as i64;
        self.quantity += match side {
            Side::Buy => delta,
            Side::Sell => -delta,
        };
    }
}

use bullet_core::Price;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

pub fn threshold_signal(price: Price, threshold: Price) -> Signal {
    if price > threshold {
        Signal::Buy
    } else if price < threshold {
        Signal::Sell
    } else {
        Signal::Hold
    }
}

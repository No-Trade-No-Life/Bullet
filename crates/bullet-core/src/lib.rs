//! Shared domain types for the Bullet event stream.

use std::fmt;

pub type OrderId = u64;
pub type Sequence = u64;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Instrument(String);

impl Instrument {
    pub fn new(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        (!value.is_empty()).then_some(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Instrument {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Price(f64);

impl Price {
    pub fn new(value: f64) -> Option<Self> {
        (value.is_finite() && value > 0.0).then_some(Self(value))
    }

    pub fn value(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Quantity(u64);

impl Quantity {
    pub fn new(value: u64) -> Option<Self> {
        (value > 0).then_some(Self(value))
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    pub id: OrderId,
    pub instrument: Instrument,
    pub side: Side,
    pub quantity: Quantity,
}

impl Order {
    pub fn new(id: OrderId, instrument: Instrument, side: Side, quantity: Quantity) -> Self {
        Self {
            id,
            instrument,
            side,
            quantity,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MarketTick {
    pub instrument: Instrument,
    pub price: Price,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Fill {
    pub order_id: OrderId,
    pub price: Price,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bar {
    pub timestamp_ns: u64,
    pub open: Price,
    pub close: Price,
}

impl Bar {
    pub fn new(timestamp_ns: u64, open: Price, close: Price) -> Self {
        Self {
            timestamp_ns,
            open,
            close,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    MarketTick(MarketTick),
    OrderSubmitted(Order),
    OrderFilled(Fill),
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventEnvelope<T> {
    pub sequence: Sequence,
    pub timestamp_ns: u64,
    pub payload: T,
}

impl<T> EventEnvelope<T> {
    pub fn new(sequence: Sequence, timestamp_ns: u64, payload: T) -> Self {
        Self {
            sequence,
            timestamp_ns,
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Bar, Event, EventEnvelope, Instrument, Order, Price, Quantity, Side};

    #[test]
    fn domain_values_require_valid_input() {
        assert_eq!(Instrument::new(""), None);
        assert_eq!(Price::new(0.0), None);
        assert_eq!(Price::new(f64::NAN), None);
        assert_eq!(Quantity::new(0), None);
    }

    #[test]
    fn order_event_keeps_its_domain_values() {
        let instrument = Instrument::new("AAPL").expect("a symbol is non-empty");
        let quantity = Quantity::new(10).expect("quantity is non-zero");
        let order = Order::new(7, instrument, Side::Buy, quantity);
        let envelope = EventEnvelope::new(3, 42, Event::OrderSubmitted(order.clone()));

        assert_eq!(envelope.sequence, 3);
        assert_eq!(envelope.timestamp_ns, 42);
        assert_eq!(envelope.payload, Event::OrderSubmitted(order));
    }

    #[test]
    fn bar_keeps_validated_prices() {
        let open = Price::new(100.25).expect("price is positive and finite");
        let close = Price::new(101.5).expect("price is positive and finite");
        let bar = Bar::new(7, open, close);

        assert_eq!(bar.timestamp_ns, 7);
        assert_eq!(bar.open.value(), 100.25);
        assert_eq!(bar.close.value(), 101.5);
    }
}

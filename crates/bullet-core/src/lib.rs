pub type Price = i64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EventKind {
    BarOpen { price: Price },
    BarClose { price: Price },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MarketEvent {
    pub sequence: u64,
    pub timestamp_ns: i128,
    pub instrument: String,
    pub kind: EventKind,
}

//! Deterministic in-memory dispatch and order-state reduction.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use bullet_core::{Event, EventEnvelope, Fill, Instrument, MarketTick, Order, OrderId, Sequence};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EventDispatcher {
    events: Vec<EventEnvelope<Event>>,
}

impl EventDispatcher {
    pub fn next_sequence(&self) -> Sequence {
        self.events.len() as Sequence + 1
    }

    pub fn envelope(&self, timestamp_ns: u64, payload: Event) -> EventEnvelope<Event> {
        EventEnvelope::new(self.next_sequence(), timestamp_ns, payload)
    }

    pub fn record(&mut self, event: EventEnvelope<Event>) {
        self.events.push(event);
    }

    pub fn events(&self) -> &[EventEnvelope<Event>] {
        &self.events
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OrderState {
    Pending(Order),
    Filled { order: Order, fill: Fill },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OrderBook {
    orders: BTreeMap<OrderId, OrderState>,
}

impl OrderBook {
    pub fn apply(&mut self, event: &EventEnvelope<Event>) -> Result<(), ReducerError> {
        match &event.payload {
            Event::MarketTick(_) => Ok(()),
            Event::OrderSubmitted(order) => self.insert_order(order.clone()),
            Event::OrderFilled(fill) => self.fill_order(fill.clone()),
        }
    }

    pub fn state(&self, order_id: OrderId) -> Option<&OrderState> {
        self.orders.get(&order_id)
    }

    fn pending_for(&self, instrument: &Instrument) -> Vec<Order> {
        self.orders
            .values()
            .filter_map(|state| match state {
                OrderState::Pending(order) if order.instrument == *instrument => {
                    Some(order.clone())
                }
                OrderState::Pending(_) | OrderState::Filled { .. } => None,
            })
            .collect()
    }

    fn insert_order(&mut self, order: Order) -> Result<(), ReducerError> {
        if self.orders.contains_key(&order.id) {
            return Err(ReducerError::DuplicateOrder(order.id));
        }

        self.orders.insert(order.id, OrderState::Pending(order));
        Ok(())
    }

    fn fill_order(&mut self, fill: Fill) -> Result<(), ReducerError> {
        let state = self
            .orders
            .get_mut(&fill.order_id)
            .ok_or(ReducerError::UnknownOrder(fill.order_id))?;
        let OrderState::Pending(order) = state else {
            return Err(ReducerError::OrderAlreadyFilled(fill.order_id));
        };
        *state = OrderState::Filled {
            order: order.clone(),
            fill,
        };
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReducerError {
    DuplicateOrder(OrderId),
    UnknownOrder(OrderId),
    OrderAlreadyFilled(OrderId),
}

impl fmt::Display for ReducerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateOrder(order_id) => write!(formatter, "order {order_id} already exists"),
            Self::UnknownOrder(order_id) => write!(formatter, "order {order_id} does not exist"),
            Self::OrderAlreadyFilled(order_id) => {
                write!(formatter, "order {order_id} is already filled")
            }
        }
    }
}

impl Error for ReducerError {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Engine {
    dispatcher: EventDispatcher,
    orders: OrderBook,
}

impl Engine {
    pub fn dispatch_at(
        &mut self,
        timestamp_ns: u64,
        payload: Event,
    ) -> Result<EventEnvelope<Event>, ReducerError> {
        let event = self.dispatcher.envelope(timestamp_ns, payload);
        self.orders.apply(&event)?;
        self.dispatcher.record(event.clone());
        Ok(event)
    }

    /// Records a market tick, then fills each earlier pending order for that instrument.
    ///
    /// A `BTreeMap` keeps the generated fill order stable by ascending order ID. The incoming
    /// tick is always recorded before any generated fill, so an order can only fill on a price
    /// published after that order entered the event stream.
    pub fn execute_market_tick_at(
        &mut self,
        timestamp_ns: u64,
        tick: MarketTick,
    ) -> Result<Vec<EventEnvelope<Event>>, ReducerError> {
        let pending_orders = self.orders.pending_for(&tick.instrument);
        let mut events = vec![self.dispatch_at(timestamp_ns, Event::MarketTick(tick.clone()))?];

        for order in pending_orders {
            events.push(self.dispatch_at(
                timestamp_ns,
                Event::OrderFilled(Fill {
                    order_id: order.id,
                    price: tick.price,
                }),
            )?);
        }

        Ok(events)
    }

    pub fn events(&self) -> &[EventEnvelope<Event>] {
        self.dispatcher.events()
    }

    pub fn order_state(&self, order_id: OrderId) -> Option<&OrderState> {
        self.orders.state(order_id)
    }
}

#[cfg(test)]
mod tests {
    use bullet_core::{Event, Fill, Instrument, MarketTick, Order, Price, Quantity, Side};

    use super::{Engine, OrderState, ReducerError};

    fn instrument(value: &str) -> Instrument {
        Instrument::new(value).expect("a symbol is non-empty")
    }

    fn order(id: u64, symbol: &str) -> Order {
        Order::new(
            id,
            instrument(symbol),
            Side::Buy,
            Quantity::new(5).expect("quantity is non-zero"),
        )
    }

    fn tick(symbol: &str, price: u64) -> MarketTick {
        MarketTick {
            instrument: instrument(symbol),
            price: Price::new(price).expect("price is non-zero"),
        }
    }

    #[test]
    fn engine_assigns_sequence_and_reduces_an_order_to_filled() {
        let mut engine = Engine::default();
        let submitted = engine
            .dispatch_at(10, Event::OrderSubmitted(order(1, "AAPL")))
            .expect("new order is accepted");
        let filled = engine
            .dispatch_at(
                20,
                Event::OrderFilled(Fill {
                    order_id: 1,
                    price: Price::new(189).expect("price is non-zero"),
                }),
            )
            .expect("pending order can fill");

        assert_eq!(submitted.sequence, 1);
        assert_eq!(filled.sequence, 2);
        assert_eq!(engine.events().len(), 2);
        assert!(matches!(
            engine.order_state(1),
            Some(OrderState::Filled { fill, .. }) if fill.price.value() == 189
        ));
    }

    #[test]
    fn market_tick_fills_only_earlier_pending_orders_for_its_instrument() {
        let mut engine = Engine::default();
        engine
            .dispatch_at(10, Event::OrderSubmitted(order(2, "MSFT")))
            .expect("new order is accepted");
        engine
            .dispatch_at(11, Event::OrderSubmitted(order(1, "AAPL")))
            .expect("new order is accepted");

        let generated = engine
            .execute_market_tick_at(20, tick("AAPL", 189))
            .expect("a valid tick executes matching pending orders");

        assert_eq!(generated.len(), 2);
        assert_eq!(generated[0].sequence, 3);
        assert_eq!(generated[0].payload, Event::MarketTick(tick("AAPL", 189)));
        assert_eq!(generated[1].sequence, 4);
        assert_eq!(
            generated[1].payload,
            Event::OrderFilled(Fill {
                order_id: 1,
                price: Price::new(189).expect("price is non-zero"),
            })
        );
        assert!(matches!(
            engine.order_state(1),
            Some(OrderState::Filled { fill, .. }) if fill.price.value() == 189
        ));
        assert!(matches!(
            engine.order_state(2),
            Some(OrderState::Pending(_))
        ));
    }

    #[test]
    fn reducer_rejects_an_unknown_fill_without_recording_it() {
        let mut engine = Engine::default();
        let error = engine
            .dispatch_at(
                10,
                Event::OrderFilled(Fill {
                    order_id: 99,
                    price: Price::new(189).expect("price is non-zero"),
                }),
            )
            .expect_err("unknown order cannot fill");

        assert_eq!(error, ReducerError::UnknownOrder(99));
        assert!(engine.events().is_empty());
    }
}

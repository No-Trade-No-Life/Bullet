use bullet_core::{Event, Fill, Instrument, Order, Price, Quantity, Side};
use bullet_engine::Engine;

fn main() {
    let instrument = Instrument::new("AAPL").expect("demo symbol is non-empty");
    let quantity = Quantity::new(10).expect("demo quantity is non-zero");
    let price = Price::new(189).expect("demo price is non-zero");
    let order = Order::new(1, instrument, Side::Buy, quantity);
    let mut engine = Engine::default();

    let submitted = engine
        .dispatch_at(1, Event::OrderSubmitted(order))
        .expect("a new demo order is accepted");
    let filled = engine
        .dispatch_at(2, Event::OrderFilled(Fill { order_id: 1, price }))
        .expect("the pending demo order fills");

    println!(
        "order 1: submitted at sequence {}, filled at sequence {}",
        submitted.sequence, filled.sequence
    );
}

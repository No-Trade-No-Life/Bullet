use bullet_core::{Event, Instrument, MarketTick, Order, Price, Quantity, Side};
use bullet_engine::Engine;

fn main() {
    let instrument = Instrument::new("AAPL").expect("demo symbol is non-empty");
    let quantity = Quantity::new(10).expect("demo quantity is non-zero");
    let price = Price::new(189).expect("demo price is non-zero");
    let order = Order::new(1, instrument.clone(), Side::Buy, quantity);
    let mut engine = Engine::default();

    let submitted = engine
        .dispatch_at(1, Event::OrderSubmitted(order))
        .expect("a new demo order is accepted");
    let executed = engine
        .execute_market_tick_at(2, MarketTick { instrument, price })
        .expect("the demo tick executes the pending order");
    let fill = executed
        .last()
        .expect("a pending demo order produces one fill");

    println!(
        "order 1: submitted at sequence {}, filled at sequence {}",
        submitted.sequence, fill.sequence
    );
}

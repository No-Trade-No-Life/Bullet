use bullet_execution::{Order, Side, fill_at_next_open};

fn main() {
    let order = Order {
        id: 1,
        created_sequence: 0,
        side: Side::Buy,
        quantity: 1,
    };
    let fill = fill_at_next_open(order, 1, 100).expect("the next open is causally eligible");
    println!(
        "Bullet ready: order {} fills at sequence {}.",
        fill.order_id, fill.sequence
    );
}

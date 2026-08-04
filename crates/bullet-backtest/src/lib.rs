use bullet_execution::{ExecutionError, Fill, Order, fill_at_next_open};

pub fn execute_next_open(
    order: Order,
    next_open_sequence: u64,
    next_open_price: i64,
) -> Result<Fill, ExecutionError> {
    fill_at_next_open(order, next_open_sequence, next_open_price)
}

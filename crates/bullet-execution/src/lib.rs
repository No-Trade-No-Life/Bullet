use bullet_core::Price;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Order {
    pub id: u64,
    pub created_sequence: u64,
    pub side: Side,
    pub quantity: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fill {
    pub order_id: u64,
    pub sequence: u64,
    pub price: Price,
    pub quantity: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionError {
    EmptyOrder,
    NonFutureOpen {
        created_sequence: u64,
        open_sequence: u64,
    },
}

pub fn fill_at_next_open(
    order: Order,
    open_sequence: u64,
    open_price: Price,
) -> Result<Fill, ExecutionError> {
    if order.quantity == 0 {
        return Err(ExecutionError::EmptyOrder);
    }
    if open_sequence <= order.created_sequence {
        return Err(ExecutionError::NonFutureOpen {
            created_sequence: order.created_sequence,
            open_sequence,
        });
    }
    Ok(Fill {
        order_id: order.id,
        sequence: open_sequence,
        price: open_price,
        quantity: order.quantity,
    })
}

#[cfg(test)]
mod tests {
    use super::{ExecutionError, Order, Side, fill_at_next_open};

    #[test]
    fn fills_only_at_a_later_open() {
        let order = Order {
            id: 7,
            created_sequence: 4,
            side: Side::Buy,
            quantity: 3,
        };
        assert_eq!(
            fill_at_next_open(order, 4, 100),
            Err(ExecutionError::NonFutureOpen {
                created_sequence: 4,
                open_sequence: 4,
            })
        );
        assert_eq!(fill_at_next_open(order, 5, 101).unwrap().price, 101);
    }
}

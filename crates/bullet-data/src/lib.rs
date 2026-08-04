use bullet_core::Price;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bar {
    pub open: Price,
    pub high: Price,
    pub low: Price,
    pub close: Price,
    pub volume: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarError {
    InvalidRange,
}

impl Bar {
    pub fn new(
        open: Price,
        high: Price,
        low: Price,
        close: Price,
        volume: u64,
    ) -> Result<Self, BarError> {
        if low > high || open < low || open > high || close < low || close > high {
            return Err(BarError::InvalidRange);
        }
        Ok(Self {
            open,
            high,
            low,
            close,
            volume,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Bar, BarError};

    #[test]
    fn rejects_close_outside_range() {
        assert_eq!(Bar::new(10, 12, 9, 13, 1), Err(BarError::InvalidRange));
    }
}

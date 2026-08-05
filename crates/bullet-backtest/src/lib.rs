//! A deterministic, long-flat dual-moving-average backtest.

use std::error::Error;
use std::fmt;

use bullet_core::{Bar, Event, Instrument, MarketTick, Order, Quantity, Side};
use bullet_engine::{Engine, OrderState, ReducerError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DualMovingAverage {
    pub fast_window: usize,
    pub slow_window: usize,
}

impl DualMovingAverage {
    pub fn new(fast_window: usize, slow_window: usize) -> Result<Self, BacktestError> {
        if fast_window == 0 || slow_window == 0 {
            return Err(BacktestError::ZeroWindow);
        }
        if fast_window >= slow_window {
            return Err(BacktestError::InvalidWindows {
                fast_window,
                slow_window,
            });
        }

        Ok(Self {
            fast_window,
            slow_window,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BacktestResult {
    pub bars: usize,
    pub fills: usize,
    pub ending_position: u64,
    pub realized_pnl: f64,
    pub mark_to_market_pnl: f64,
}

pub fn run_dual_moving_average(
    bars: &[Bar],
    strategy: DualMovingAverage,
    quantity: Quantity,
) -> Result<BacktestResult, BacktestError> {
    let instrument = Instrument::new("PARQUET").expect("constant instrument is non-empty");
    let mut engine = Engine::default();
    let mut closes = Vec::with_capacity(bars.len());
    let mut position = 0_u64;
    let mut realized_pnl = 0.0;
    let mut fills = 0;
    let mut next_order_id = 1;

    for bar in bars {
        apply_open(
            &mut engine,
            &instrument,
            bar,
            &mut position,
            &mut realized_pnl,
            &mut fills,
        )?;
        closes.push(bar.close.value());

        let Some(signal) = signal(&closes, strategy) else {
            continue;
        };
        let side = match (signal, position) {
            (Signal::Long, 0) => Some(Side::Buy),
            (Signal::Flat, value) if value > 0 => Some(Side::Sell),
            (Signal::Long, _) | (Signal::Flat, _) => None,
        };
        let Some(side) = side else {
            continue;
        };
        let order = Order::new(next_order_id, instrument.clone(), side, quantity);
        engine
            .dispatch_at(bar.timestamp_ns, Event::OrderSubmitted(order))
            .map_err(BacktestError::Engine)?;
        next_order_id += 1;
    }

    let mark_to_market_pnl = bars.last().map_or(realized_pnl, |bar| {
        realized_pnl + position as f64 * bar.close.value()
    });

    Ok(BacktestResult {
        bars: bars.len(),
        fills,
        ending_position: position,
        realized_pnl,
        mark_to_market_pnl,
    })
}

fn apply_open(
    engine: &mut Engine,
    instrument: &Instrument,
    bar: &Bar,
    position: &mut u64,
    realized_pnl: &mut f64,
    fills: &mut usize,
) -> Result<(), BacktestError> {
    let generated = engine
        .execute_market_tick_at(
            bar.timestamp_ns,
            MarketTick {
                instrument: instrument.clone(),
                price: bar.open,
            },
        )
        .map_err(BacktestError::Engine)?;

    for event in generated {
        let Event::OrderFilled(fill) = event.payload else {
            continue;
        };
        let Some(OrderState::Filled { order, .. }) = engine.order_state(fill.order_id) else {
            return Err(BacktestError::MissingFilledOrder(fill.order_id));
        };
        let value = order.quantity.value() as f64 * fill.price.value();
        match order.side {
            Side::Buy => {
                *position += order.quantity.value();
                *realized_pnl -= value;
            }
            Side::Sell => {
                *position -= order.quantity.value();
                *realized_pnl += value;
            }
        }
        *fills += 1;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Signal {
    Long,
    Flat,
}

fn signal(closes: &[f64], strategy: DualMovingAverage) -> Option<Signal> {
    if closes.len() < strategy.slow_window {
        return None;
    }

    let fast = average(&closes[closes.len() - strategy.fast_window..]);
    let slow = average(&closes[closes.len() - strategy.slow_window..]);
    Some(if fast > slow {
        Signal::Long
    } else {
        Signal::Flat
    })
}

fn average(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[derive(Debug)]
pub enum BacktestError {
    ZeroWindow,
    InvalidWindows {
        fast_window: usize,
        slow_window: usize,
    },
    Engine(ReducerError),
    MissingFilledOrder(u64),
}

impl fmt::Display for BacktestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWindow => formatter.write_str("moving-average windows must be non-zero"),
            Self::InvalidWindows {
                fast_window,
                slow_window,
            } => write!(
                formatter,
                "fast window {fast_window} must be smaller than slow window {slow_window}"
            ),
            Self::Engine(error) => write!(formatter, "engine rejected backtest event: {error}"),
            Self::MissingFilledOrder(order_id) => {
                write!(formatter, "engine did not retain filled order {order_id}")
            }
        }
    }
}

impl Error for BacktestError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Engine(error) => Some(error),
            Self::ZeroWindow | Self::InvalidWindows { .. } | Self::MissingFilledOrder(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use bullet_core::{Bar, Price, Quantity};

    use super::{DualMovingAverage, run_dual_moving_average};

    fn bar(timestamp_ns: u64, open: f64, close: f64) -> Bar {
        Bar::new(
            timestamp_ns,
            Price::new(open).expect("test open is valid"),
            Price::new(close).expect("test close is valid"),
        )
    }

    #[test]
    fn long_flat_strategy_fills_at_the_next_bar_open() {
        let bars = [
            bar(1, 10.0, 10.0),
            bar(2, 11.0, 11.0),
            bar(3, 12.0, 12.0),
            bar(4, 10.0, 10.0),
            bar(5, 13.0, 9.0),
        ];
        let strategy = DualMovingAverage::new(2, 3).expect("windows are valid");
        let quantity = Quantity::new(1).expect("quantity is valid");

        let result = run_dual_moving_average(&bars, strategy, quantity)
            .expect("backtest runs on valid bars");

        assert_eq!(result.bars, 5);
        assert_eq!(result.fills, 2);
        assert_eq!(result.ending_position, 0);
        assert_eq!(result.realized_pnl, 3.0);
        assert_eq!(result.mark_to_market_pnl, 3.0);
    }

    #[test]
    fn strategy_requires_fast_window_smaller_than_slow_window() {
        assert!(DualMovingAverage::new(3, 3).is_err());
        assert!(DualMovingAverage::new(0, 3).is_err());
    }
}

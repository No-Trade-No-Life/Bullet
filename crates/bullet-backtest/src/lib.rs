//! A deterministic, long-flat dual-moving-average backtest and its evaluation metrics.

use std::error::Error;
use std::fmt;

use bullet_core::{Bar, Event, Instrument, MarketTick, Order, Quantity, Side};
use bullet_engine::{Engine, OrderState, ReducerError};

const NANOS_PER_DAY: u64 = 86_400_000_000_000;
const NANOS_PER_YEAR: f64 = 365.25 * NANOS_PER_DAY as f64;
const TRADING_DAYS_PER_YEAR: f64 = 252.0;

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
    pub round_trips: usize,
    pub ending_position: u64,
    pub realized_pnl: f64,
    pub mark_to_market_pnl: f64,
    pub performance: Performance,
}

/// Metrics are derived from daily UTC close equity with zero risk-free rate.
///
/// Initial capital equals the first bar close multiplied by the configured quantity. This models
/// a fully funded one-unit position, not a futures margin account or contract multiplier.
#[derive(Clone, Debug, PartialEq)]
pub struct Performance {
    pub initial_equity: f64,
    pub final_equity: f64,
    pub cumulative_return: f64,
    pub annualized_return: Option<f64>,
    pub annualized_sharpe: Option<f64>,
    pub max_drawdown: f64,
    pub return_drawdown_ratio: Option<f64>,
    pub daily_observations: usize,
}

pub fn run_dual_moving_average(
    bars: &[Bar],
    strategy: DualMovingAverage,
    quantity: Quantity,
) -> Result<BacktestResult, BacktestError> {
    let first_bar = bars.first().ok_or(BacktestError::EmptyBars)?;
    let initial_equity = first_bar.close.value() * quantity.value() as f64;
    let instrument = Instrument::new("PARQUET").expect("constant instrument is non-empty");
    let mut engine = Engine::default();
    let mut closes = Vec::with_capacity(bars.len());
    let mut position = 0_u64;
    let mut realized_pnl = 0.0;
    let mut fills = 0;
    let mut round_trips = 0;
    let mut next_order_id = 1;
    let mut daily_equity = Vec::new();

    for bar in bars {
        apply_open(
            &mut engine,
            &instrument,
            bar,
            &mut position,
            &mut realized_pnl,
            &mut fills,
            &mut round_trips,
        )?;
        closes.push(bar.close.value());
        record_daily_equity(
            &mut daily_equity,
            bar.timestamp_ns,
            initial_equity + realized_pnl + position as f64 * bar.close.value(),
        );

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
        round_trips,
        ending_position: position,
        realized_pnl,
        mark_to_market_pnl,
        performance: evaluate(
            &daily_equity,
            initial_equity,
            bars[0].timestamp_ns,
            bars[bars.len() - 1].timestamp_ns,
        )?,
    })
}

fn apply_open(
    engine: &mut Engine,
    instrument: &Instrument,
    bar: &Bar,
    position: &mut u64,
    realized_pnl: &mut f64,
    fills: &mut usize,
    round_trips: &mut usize,
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
                *round_trips += 1;
            }
        }
        *fills += 1;
    }

    Ok(())
}

fn record_daily_equity(daily_equity: &mut Vec<(u64, f64)>, timestamp_ns: u64, equity: f64) {
    let day = timestamp_ns / NANOS_PER_DAY;
    if let Some((last_day, last_equity)) = daily_equity.last_mut()
        && *last_day == day
    {
        *last_equity = equity;
        return;
    }
    daily_equity.push((day, equity));
}

fn evaluate(
    daily_equity: &[(u64, f64)],
    initial_equity: f64,
    first_timestamp_ns: u64,
    last_timestamp_ns: u64,
) -> Result<Performance, BacktestError> {
    let final_equity = daily_equity
        .last()
        .map(|(_, equity)| *equity)
        .ok_or(BacktestError::EmptyBars)?;
    let cumulative_return = final_equity / initial_equity - 1.0;
    let annualized_return = annualized_return(
        initial_equity,
        final_equity,
        last_timestamp_ns.saturating_sub(first_timestamp_ns),
    );
    let max_drawdown = max_drawdown(daily_equity);
    let return_drawdown_ratio = annualized_return
        .filter(|_| max_drawdown < 0.0)
        .map(|value| value / -max_drawdown);

    Ok(Performance {
        initial_equity,
        final_equity,
        cumulative_return,
        annualized_return,
        annualized_sharpe: annualized_sharpe(initial_equity, daily_equity),
        max_drawdown,
        return_drawdown_ratio,
        daily_observations: daily_equity.len(),
    })
}

fn annualized_return(initial_equity: f64, final_equity: f64, elapsed_ns: u64) -> Option<f64> {
    (elapsed_ns > 0)
        .then(|| (final_equity / initial_equity).powf(NANOS_PER_YEAR / elapsed_ns as f64) - 1.0)
}

fn annualized_sharpe(initial_equity: f64, daily_equity: &[(u64, f64)]) -> Option<f64> {
    let mut returns = Vec::with_capacity(daily_equity.len());
    let mut previous_equity = initial_equity;
    for (_, equity) in daily_equity {
        returns.push(*equity / previous_equity - 1.0);
        previous_equity = *equity;
    }
    if returns.len() < 2 {
        return None;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;
    (variance > 0.0).then(|| mean / variance.sqrt() * TRADING_DAYS_PER_YEAR.sqrt())
}

fn max_drawdown(daily_equity: &[(u64, f64)]) -> f64 {
    daily_equity
        .iter()
        .fold(
            (f64::NEG_INFINITY, 0.0_f64),
            |(peak, drawdown), (_, equity)| {
                let peak = peak.max(*equity);
                (peak, drawdown.min(*equity / peak - 1.0))
            },
        )
        .1
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
    EmptyBars,
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
            Self::EmptyBars => formatter.write_str("backtest requires at least one bar"),
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
            Self::EmptyBars
            | Self::ZeroWindow
            | Self::InvalidWindows { .. }
            | Self::MissingFilledOrder(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use bullet_core::{Bar, Price, Quantity};

    use super::{DualMovingAverage, run_dual_moving_average};

    const DAY: u64 = 86_400_000_000_000;

    fn bar(day: u64, open: f64, close: f64) -> Bar {
        Bar::new(
            day * DAY,
            Price::new(open).expect("test open is valid"),
            Price::new(close).expect("test close is valid"),
        )
    }

    #[test]
    fn backtest_evaluates_next_open_execution_and_daily_performance() {
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
        assert_eq!(result.round_trips, 1);
        assert_eq!(result.ending_position, 0);
        assert_eq!(result.realized_pnl, 3.0);
        assert_eq!(result.mark_to_market_pnl, 3.0);
        assert_eq!(result.performance.initial_equity, 10.0);
        assert_eq!(result.performance.final_equity, 13.0);
        assert!((result.performance.cumulative_return - 0.3).abs() < f64::EPSILON);
        assert_eq!(result.performance.max_drawdown, 0.0);
        assert_eq!(result.performance.daily_observations, 5);
        assert_eq!(result.performance.return_drawdown_ratio, None);
    }

    #[test]
    fn strategy_requires_fast_window_smaller_than_slow_window() {
        assert!(DualMovingAverage::new(3, 3).is_err());
        assert!(DualMovingAverage::new(0, 3).is_err());
    }
}

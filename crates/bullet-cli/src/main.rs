use std::error::Error;
use std::path::PathBuf;

use bullet_backtest::{DualMovingAverage, run_dual_moving_average};
use bullet_core::Quantity;
use bullet_data::read_bars;

fn main() {
    if let Err(error) = run() {
        eprintln!("bullet: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse(std::env::args().skip(1))?;
    let bars = read_bars(arguments.path)?;
    let strategy = DualMovingAverage::new(arguments.fast_window, arguments.slow_window)?;
    let quantity = Quantity::new(1).expect("constant quantity is non-zero");
    let result = run_dual_moving_average(&bars, strategy, quantity)?;

    println!("bars: {}", result.bars);
    println!("fills: {}", result.fills);
    println!("ending_position: {}", result.ending_position);
    println!("realized_pnl: {:.6}", result.realized_pnl);
    println!("mark_to_market_pnl: {:.6}", result.mark_to_market_pnl);
    Ok(())
}

struct Arguments {
    path: PathBuf,
    fast_window: usize,
    slow_window: usize,
}

impl Arguments {
    fn parse(mut values: impl Iterator<Item = String>) -> Result<Self, ArgumentError> {
        let path = values.next().ok_or(ArgumentError::Usage)?;
        let fast_window = values
            .next()
            .map(|value| value.parse())
            .transpose()
            .map_err(ArgumentError::InvalidFastWindow)?
            .unwrap_or(20);
        let slow_window = values
            .next()
            .map(|value| value.parse())
            .transpose()
            .map_err(ArgumentError::InvalidSlowWindow)?
            .unwrap_or(50);
        if values.next().is_some() {
            return Err(ArgumentError::Usage);
        }

        Ok(Self {
            path: PathBuf::from(path),
            fast_window,
            slow_window,
        })
    }
}

#[derive(Debug)]
enum ArgumentError {
    Usage,
    InvalidFastWindow(std::num::ParseIntError),
    InvalidSlowWindow(std::num::ParseIntError),
}

impl std::fmt::Display for ArgumentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage => {
                formatter.write_str("usage: bullet-cli <bars.parquet> [fast-window] [slow-window]")
            }
            Self::InvalidFastWindow(error) => write!(formatter, "invalid fast window: {error}"),
            Self::InvalidSlowWindow(error) => write!(formatter, "invalid slow window: {error}"),
        }
    }
}

impl Error for ArgumentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidFastWindow(error) => Some(error),
            Self::InvalidSlowWindow(error) => Some(error),
            Self::Usage => None,
        }
    }
}

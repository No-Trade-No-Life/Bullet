use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use bullet_backtest::{DualMovingAverage, Performance, run_dual_moving_average};
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
    let data_size_bytes = std::fs::metadata(&arguments.path)?.len();
    let started_at = Instant::now();
    let bars = read_bars(&arguments.path)?;
    let strategy = DualMovingAverage::new(arguments.fast_window, arguments.slow_window)?;
    let quantity = Quantity::new(1).expect("constant quantity is non-zero");
    let result = run_dual_moving_average(&bars, strategy, quantity)?;
    let elapsed = started_at.elapsed();

    println!("bars: {}", result.bars);
    println!("data_size_bytes: {data_size_bytes}");
    println!("runtime_ms: {}", elapsed.as_millis());
    println!("peak_rss_bytes: {}", peak_rss_bytes());
    println!("fills: {}", result.fills);
    println!("round_trips: {}", result.round_trips);
    println!("ending_position: {}", result.ending_position);
    println!("realized_pnl: {:.6}", result.realized_pnl);
    println!("mark_to_market_pnl: {:.6}", result.mark_to_market_pnl);
    print_performance(&result.performance);
    Ok(())
}

fn print_performance(performance: &Performance) {
    println!("initial_equity: {:.6}", performance.initial_equity);
    println!("final_equity: {:.6}", performance.final_equity);
    println!("cumulative_return: {:.6}", performance.cumulative_return);
    print_optional("annualized_return", performance.annualized_return);
    print_optional("annualized_sharpe", performance.annualized_sharpe);
    println!("max_drawdown: {:.6}", performance.max_drawdown);
    print_optional("return_drawdown_ratio", performance.return_drawdown_ratio);
    println!("daily_observations: {}", performance.daily_observations);
}

fn print_optional(label: &str, value: Option<f64>) {
    match value {
        Some(value) => println!("{label}: {value:.6}"),
        None => println!("{label}: n/a"),
    }
}

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` points to valid writable storage, and `RUSAGE_SELF` queries this process.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return 0;
    }
    let maximum_rss = unsafe { usage.assume_init() }.ru_maxrss as u64;

    #[cfg(target_os = "macos")]
    {
        maximum_rss
    }
    #[cfg(not(target_os = "macos"))]
    {
        maximum_rss * 1024
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
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

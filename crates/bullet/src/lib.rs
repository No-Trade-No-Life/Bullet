//! Stable public API for Bullet strategy source files.

use std::error::Error;
use std::path::Path;

pub use bullet_backtest::{BarContext, Config, Order, Strategy};
pub use bullet_core::Bar;

pub fn run(path: impl AsRef<Path>, strategy: &mut impl Strategy) -> Result<(), Box<dyn Error>> {
    let config = Config::read(path)?;
    let result = bullet_backtest::run(&config, strategy)?;
    println!("bars: {}", result.bars);
    println!("data_size_bytes: {}", result.data_size_bytes);
    println!("runtime_ms: {}", result.runtime_ms);
    println!("peak_rss_bytes: {}", peak_rss_bytes());
    println!("fills: {}", result.fills);
    println!("round_trips: {}", result.round_trips);
    println!("fees_paid: {:.6}", result.fees_paid);
    println!("slippage_paid: {:.6}", result.slippage_paid);
    for (instrument, position) in &result.ending_positions {
        println!("ending_position.{instrument}: {position}");
    }
    let value = &result.performance;
    println!("initial_equity: {:.6}", value.initial_equity);
    println!("final_equity: {:.6}", value.final_equity);
    println!("cumulative_return: {:.6}", value.cumulative_return);
    optional("annualized_return", value.annualized_return);
    optional("annualized_sharpe", value.annualized_sharpe);
    println!("max_drawdown: {:.6}", value.max_drawdown);
    optional("return_drawdown_ratio", value.return_drawdown_ratio);
    println!("daily_observations: {}", value.daily_observations);
    Ok(())
}

fn optional(label: &str, value: Option<f64>) {
    match value {
        Some(value) => println!("{label}: {value:.6}"),
        None => println!("{label}: n/a"),
    }
}

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: storage is valid and RUSAGE_SELF reads the current process only.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return 0;
    }
    let maximum = unsafe { usage.assume_init() }.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    {
        maximum
    }
    #[cfg(not(target_os = "macos"))]
    {
        maximum * 1024
    }
}
#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

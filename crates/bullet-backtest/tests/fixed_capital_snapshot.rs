use std::path::PathBuf;

use bullet_backtest::fixed_capital::{
    FixedCapitalConfig, FixedCapitalContext, FixedCapitalInstrument, FixedCapitalStrategy,
    TimestampInterpretation, TradingDay, run_fixed_capital,
};

struct NoOrders;

impl FixedCapitalStrategy for NoOrders {
    fn on_timestamp(
        &mut self,
        _context: FixedCapitalContext<'_>,
    ) -> Result<Vec<bullet_backtest::fixed_capital::ExposureOrder>, String> {
        Ok(Vec::new())
    }
}

#[test]
#[ignore = "requires BULLET_FIXED_CAPITAL_SNAPSHOT with IF/IH/IC/IM 8888 Parquet files"]
fn replays_the_explicit_four_index_snapshot_with_naive_shanghai_wall_clocks() {
    let root = PathBuf::from(
        std::env::var_os("BULLET_FIXED_CAPITAL_SNAPSHOT")
            .expect("BULLET_FIXED_CAPITAL_SNAPSHOT is required"),
    );
    let available = [
        ("IF", 991_320_usize),
        ("IH", 664_350),
        ("IC", 664_350),
        ("IM", 234_000),
    ];
    let count = std::env::var("BULLET_FIXED_CAPITAL_INSTRUMENTS")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("instrument count is an integer")
        })
        .unwrap_or(available.len());
    assert!((1..=available.len()).contains(&count));
    let instruments = available[..count]
        .iter()
        .map(|(id, _)| *id)
        .map(|id| FixedCapitalInstrument {
            id: id.to_owned(),
            data: root.join(format!("{id}8888.parquet")),
            timestamp_interpretation: TimestampInterpretation::NaiveAsiaShanghaiWallClock,
        })
        .collect();
    let config = FixedCapitalConfig {
        instruments,
        components: vec!["fixture".to_owned()],
        evaluation_days: vec![TradingDay::new(2026, 8, 3).expect("valid fixture day")],
    };

    let result = run_fixed_capital(&config, &mut NoOrders).expect("snapshot replay succeeds");

    let expected_bars: usize = available[..count].iter().map(|(_, rows)| rows).sum();
    assert_eq!(result.bars, expected_bars);
    assert!(result.trades.is_empty());
    assert_eq!(result.daily_returns.len(), 1);
    println!(
        "PERF_PROGRESS processed={} total=2554020 elapsed_sec={:.6} unit=bars instruments={count}",
        result.bars,
        result.runtime_ms as f64 / 1000.0,
    );
}

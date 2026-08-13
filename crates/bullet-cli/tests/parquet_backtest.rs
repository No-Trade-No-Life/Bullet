use std::fs::{self, File};
use std::process::Command;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

#[test]
fn cli_runs_a_dual_moving_average_through_long_and_short_round_trips() {
    let root = std::env::temp_dir().join(format!("bullet-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("temporary directory is writable");
    let parquet = root.join("bars.parquet");
    write_bars(
        &parquet,
        &[10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 12.0],
    );
    let strategy = root.join("strategy.rs");
    fs::write(
        &strategy,
        r#"
use bullet::{BarContext, Order, Strategy};
pub struct DualMovingAverage { closes: Vec<f64> }
pub fn strategy() -> DualMovingAverage { DualMovingAverage { closes: Vec::new() } }
impl Strategy for DualMovingAverage {
    fn on_bar(&mut self, context: BarContext<'_>) -> Order {
        self.closes.push(context.bar.close.value());
        if self.closes.len() < 3 { return Order::None; }
        let fast = (&self.closes[self.closes.len() - 2..]).iter().sum::<f64>() / 2.0;
        let slow = (&self.closes[self.closes.len() - 3..]).iter().sum::<f64>() / 3.0;
        if fast > slow {
            if context.position < 0 { Order::Close }
            else if context.position == 0 { Order::Buy(1) }
            else { Order::None }
        } else if fast < slow {
            if context.position > 0 { Order::Close }
            else if context.position == 0 { Order::Sell(1) }
            else { Order::None }
        } else { Order::None }
    }
}
"#,
    )
    .expect("strategy source is writable");
    let config = root.join("backtest.toml");
    fs::write(
        &config,
        format!(
            r#"
version = 1
[backtest]
mode = "bar"
initial_cash = 100.0
currency = "CNY"
[execution]
fill_price = "next_bar_open"
slippage_bps = 0.0
[fees]
mode = "per_contract"
open = 1.0
close = 2.0
[[instruments]]
id = "TEST"
data = "{}"
multiplier = 1.0
margin_rate = 0.1
tick_size = 0.1
"#,
            parquet.display()
        ),
    )
    .expect("config is writable");

    let output = Command::new(env!("CARGO_BIN_EXE_bullet-cli"))
        .args([
            "run",
            strategy.to_str().expect("UTF-8 path"),
            "--config",
            config.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("CLI starts");
    let _ = fs::remove_dir_all(&root);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("CLI output is UTF-8");
    assert!(stdout.contains("strategy_binary: "));
    assert!(stdout.contains("bars: 9"));
    assert!(stdout.contains("fills: 4"));
    assert!(stdout.contains("round_trips: 2"));
    assert!(stdout.contains("fees_paid: 6.000000"));
    assert!(stdout.contains("ending_position.TEST: 0"));
    assert!(stdout.contains("final_equity: 90.000000"));
}

#[test]
fn cli_reports_an_order_submitted_on_the_final_bar_as_unfilled() {
    let root = std::env::temp_dir().join(format!("bullet-cli-final-order-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("temporary directory is writable");
    let parquet = root.join("bars.parquet");
    write_bars(&parquet, &[10.0, 10.0]);
    let strategy = root.join("strategy.rs");
    fs::write(
        &strategy,
        r#"
use bullet::{BarContext, Order, Strategy};
pub struct BuyEveryBar;
pub fn strategy() -> BuyEveryBar { BuyEveryBar }
impl Strategy for BuyEveryBar {
    fn on_bar(&mut self, _: BarContext<'_>) -> Order { Order::Buy(1) }
}
"#,
    )
    .expect("strategy source is writable");
    let config = root.join("backtest.toml");
    fs::write(
        &config,
        format!(
            r#"
version = 1
[backtest]
mode = "bar"
initial_cash = 100.0
currency = "CNY"
[execution]
fill_price = "next_bar_open"
slippage_bps = 0.0
[fees]
mode = "per_contract"
open = 0.0
close = 0.0
[[instruments]]
id = "TEST"
data = "{}"
multiplier = 1.0
margin_rate = 0.1
tick_size = 0.1
"#,
            parquet.display()
        ),
    )
    .expect("config is writable");

    let output = Command::new(env!("CARGO_BIN_EXE_bullet-cli"))
        .args([
            "run",
            strategy.to_str().expect("UTF-8 path"),
            "--config",
            config.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("CLI starts");
    let _ = fs::remove_dir_all(&root);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("CLI output is UTF-8");
    assert!(stdout.contains("fills: 1"));
    assert!(stdout.contains("unfilled_orders: 1"));
    assert!(stdout.contains("ending_position.TEST: 1"));
}

fn write_bars(path: &std::path::Path, prices: &[f64]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "date",
            DataType::Timestamp(
                arrow_schema::TimeUnit::Nanosecond,
                Some("Asia/Shanghai".into()),
            ),
            false,
        ),
        Field::new("open", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(
                TimestampNanosecondArray::from(
                    prices
                        .iter()
                        .enumerate()
                        .map(|(index, _)| (index as i64 + 1) * 86_400_000_000_000)
                        .collect::<Vec<_>>(),
                )
                .with_timezone("Asia/Shanghai"),
            ) as ArrayRef,
            Arc::new(Float64Array::from(prices.to_vec())) as ArrayRef,
            Arc::new(Float64Array::from(prices.to_vec())) as ArrayRef,
        ],
    )
    .expect("valid batch");
    let file = File::create(path).expect("parquet writable");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer initializes");
    writer.write(&batch).expect("writes batch");
    writer.close().expect("writes footer");
}

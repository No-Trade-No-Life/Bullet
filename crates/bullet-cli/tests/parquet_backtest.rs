use std::fs::{self, File};
use std::process::Command;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

#[test]
fn cli_compiles_and_runs_a_strategy_source_with_toml_config() {
    let root = std::env::temp_dir().join(format!("bullet-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("temporary directory is writable");
    let parquet = root.join("bars.parquet");
    write_bars(&parquet);
    let strategy = root.join("strategy.rs");
    fs::write(&strategy, r#"
use bullet::{BarContext, Order, Strategy};
pub struct BuyOnce;
pub fn strategy() -> BuyOnce { BuyOnce }
impl Strategy for BuyOnce { fn on_bar(&mut self, context: BarContext<'_>) -> Order { if context.position == 0 { Order::Buy(1) } else { Order::Close } } }
"#).expect("strategy source is writable");
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
close = 1.0
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
    assert!(stdout.contains("bars: 5"));
    assert!(stdout.contains("fills: 4"));
    assert!(stdout.contains("fees_paid: 4.000000"));
    assert!(stdout.contains("ending_position.TEST: 0"));
}

fn write_bars(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "date",
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("open", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(TimestampNanosecondArray::from(vec![
                86_400_000_000_000_i64,
                172_800_000_000_000,
                259_200_000_000_000,
                345_600_000_000_000,
                432_000_000_000_000,
            ])) as ArrayRef,
            Arc::new(Float64Array::from(vec![10.0, 11.0, 12.0, 13.0, 14.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![10.0, 11.0, 12.0, 13.0, 14.0])) as ArrayRef,
        ],
    )
    .expect("valid batch");
    let file = File::create(path).expect("parquet writable");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer initializes");
    writer.write(&batch).expect("writes batch");
    writer.close().expect("writes footer");
}

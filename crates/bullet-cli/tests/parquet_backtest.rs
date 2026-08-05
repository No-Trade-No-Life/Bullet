use std::fs::File;
use std::process::Command;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

#[test]
fn cli_runs_dual_moving_average_on_parquet_input() {
    let path = std::env::temp_dir().join(format!(
        "bullet-cli-{}-{}.parquet",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos()
    ));
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
            Arc::new(Float64Array::from(vec![10.0, 11.0, 12.0, 10.0, 13.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![10.0, 11.0, 12.0, 10.0, 9.0])) as ArrayRef,
        ],
    )
    .expect("test batch has a valid schema");
    let file = File::create(&path).expect("temporary parquet path is writable");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer initializes");
    writer.write(&batch).expect("writer accepts batch");
    writer.close().expect("writer writes footer");

    let output = Command::new(env!("CARGO_BIN_EXE_bullet-cli"))
        .arg(&path)
        .arg("2")
        .arg("3")
        .output()
        .expect("CLI starts");
    std::fs::remove_file(&path).expect("temporary parquet file is removed");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("CLI output is UTF-8");
    assert!(stdout.contains("bars: 5"));
    assert!(stdout.contains("data_size_bytes: "));
    assert!(stdout.contains("runtime_ms: "));
    assert!(stdout.contains("peak_rss_bytes: "));
    assert!(stdout.contains("fills: 2"));
    assert!(stdout.contains("ending_position: 0"));
    assert!(stdout.contains("realized_pnl: 3.000000"));
    assert!(stdout.contains("cumulative_return: 0.300000"));
    assert!(stdout.contains("max_drawdown: 0.000000"));
}

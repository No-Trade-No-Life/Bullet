//! Parquet market-bar input for Bullet backtests.

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::Path;

use arrow_array::{Array, Float64Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::ArrowError;
use bullet_core::{Bar, Price};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const DATE_COLUMN: &str = "date";
const OPEN_COLUMN: &str = "open";
const CLOSE_COLUMN: &str = "close";
const OPEN_INTEREST_COLUMN: &str = "open_interest";

/// A historical bar used to seed a live strategy before CTPD ticks take over.
///
/// `timestamp_ns` deliberately remains the source's timezone-naive market
/// timestamp so the live runner can apply the Parquet natural-day convention
/// at its splice boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryBar {
    pub timestamp_ns: u64,
    pub open: f64,
    pub close: f64,
    pub open_interest: f64,
}

pub fn read_bars(path: impl AsRef<Path>) -> Result<Vec<Bar>, DataError> {
    let file = File::open(path).map_err(DataError::Open)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(DataError::Parquet)?
        .build()
        .map_err(DataError::Parquet)?;
    let mut bars = Vec::new();

    for batch in reader {
        append_batch(&mut bars, &batch.map_err(DataError::Arrow)?)?;
    }

    validate_timestamps(&bars)?;
    Ok(bars)
}

/// Reads the newest `maximum` bars required to reconstruct the current
/// session state. This is startup work only; the live inference hot path never
/// reads Parquet.
pub fn read_history_tail(
    path: impl AsRef<Path>,
    maximum: usize,
) -> Result<Vec<HistoryBar>, DataError> {
    if maximum == 0 {
        return Err(DataError::EmptyTail);
    }
    let file = File::open(path).map_err(DataError::Open)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(DataError::Parquet)?
        .build()
        .map_err(DataError::Parquet)?;
    let mut bars = Vec::new();

    for batch in reader {
        append_history_batch(&mut bars, &batch.map_err(DataError::Arrow)?)?;
    }
    validate_history_timestamps(&bars)?;
    let start = bars.len().saturating_sub(maximum);
    Ok(bars.split_off(start))
}

fn append_batch(bars: &mut Vec<Bar>, batch: &RecordBatch) -> Result<(), DataError> {
    let dates = required_date_column(batch)?;
    let opens = required_f64_column(batch, OPEN_COLUMN)?;
    let closes = required_f64_column(batch, CLOSE_COLUMN)?;

    for row in 0..batch.num_rows() {
        let timestamp_ns = timestamp_ns(required_value(dates, row, DATE_COLUMN)?, row)?;
        let open = price(required_value(opens, row, OPEN_COLUMN)?, row, OPEN_COLUMN)?;
        let close = price(
            required_value(closes, row, CLOSE_COLUMN)?,
            row,
            CLOSE_COLUMN,
        )?;
        bars.push(Bar::new(timestamp_ns, open, close));
    }

    Ok(())
}

fn append_history_batch(bars: &mut Vec<HistoryBar>, batch: &RecordBatch) -> Result<(), DataError> {
    let dates = required_date_column(batch)?;
    let opens = required_f64_column(batch, OPEN_COLUMN)?;
    let closes = required_f64_column(batch, CLOSE_COLUMN)?;
    let open_interest = required_f64_column(batch, OPEN_INTEREST_COLUMN)?;

    for row in 0..batch.num_rows() {
        let timestamp_ns = timestamp_ns(required_value(dates, row, DATE_COLUMN)?, row)?;
        let open = finite(required_value(opens, row, OPEN_COLUMN)?, row, OPEN_COLUMN)?;
        let close = finite(
            required_value(closes, row, CLOSE_COLUMN)?,
            row,
            CLOSE_COLUMN,
        )?;
        let open_interest = finite(
            required_value(open_interest, row, OPEN_INTEREST_COLUMN)?,
            row,
            OPEN_INTEREST_COLUMN,
        )?;
        if open <= 0.0 || close <= 0.0 || open_interest < 0.0 {
            return Err(DataError::InvalidHistoryValue { row });
        }
        bars.push(HistoryBar {
            timestamp_ns,
            open,
            close,
            open_interest,
        });
    }

    Ok(())
}

fn required_date_column(batch: &RecordBatch) -> Result<&TimestampNanosecondArray, DataError> {
    batch
        .column_by_name(DATE_COLUMN)
        .ok_or(DataError::MissingColumn(DATE_COLUMN))?
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .ok_or(DataError::InvalidColumnType(DATE_COLUMN))
}

fn required_f64_column<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a Float64Array, DataError> {
    batch
        .column_by_name(name)
        .ok_or(DataError::MissingColumn(name))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(DataError::InvalidColumnType(name))
}

fn required_value<T: Copy>(
    array: &impl ArrayValue<T>,
    row: usize,
    name: &'static str,
) -> Result<T, DataError> {
    (!array.is_null(row))
        .then(|| array.value(row))
        .ok_or(DataError::NullValue { column: name, row })
}

trait ArrayValue<T>: Array {
    fn value(&self, row: usize) -> T;
}

impl ArrayValue<i64> for TimestampNanosecondArray {
    fn value(&self, row: usize) -> i64 {
        TimestampNanosecondArray::value(self, row)
    }
}

impl ArrayValue<f64> for Float64Array {
    fn value(&self, row: usize) -> f64 {
        Float64Array::value(self, row)
    }
}

fn timestamp_ns(value: i64, row: usize) -> Result<u64, DataError> {
    u64::try_from(value).map_err(|_| DataError::InvalidTimestamp { row, value })
}

fn price(value: f64, row: usize, column: &'static str) -> Result<Price, DataError> {
    Price::new(value).ok_or(DataError::InvalidPrice { column, row, value })
}

fn finite(value: f64, row: usize, column: &'static str) -> Result<f64, DataError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or(DataError::InvalidValue { column, row, value })
}

fn validate_timestamps(bars: &[Bar]) -> Result<(), DataError> {
    for (row, pair) in bars.windows(2).enumerate() {
        if pair[0].timestamp_ns >= pair[1].timestamp_ns {
            return Err(DataError::NonIncreasingTimestamp { row: row + 1 });
        }
    }

    Ok(())
}

fn validate_history_timestamps(bars: &[HistoryBar]) -> Result<(), DataError> {
    for (row, pair) in bars.windows(2).enumerate() {
        if pair[0].timestamp_ns >= pair[1].timestamp_ns {
            return Err(DataError::NonIncreasingTimestamp { row: row + 1 });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum DataError {
    Open(std::io::Error),
    Parquet(parquet::errors::ParquetError),
    Arrow(ArrowError),
    MissingColumn(&'static str),
    InvalidColumnType(&'static str),
    NullValue {
        column: &'static str,
        row: usize,
    },
    InvalidTimestamp {
        row: usize,
        value: i64,
    },
    InvalidPrice {
        column: &'static str,
        row: usize,
        value: f64,
    },
    InvalidValue {
        column: &'static str,
        row: usize,
        value: f64,
    },
    InvalidHistoryValue {
        row: usize,
    },
    EmptyTail,
    NonIncreasingTimestamp {
        row: usize,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(error) => write!(formatter, "cannot open Parquet file: {error}"),
            Self::Parquet(error) => write!(formatter, "cannot read Parquet file: {error}"),
            Self::Arrow(error) => write!(formatter, "cannot decode Parquet batch: {error}"),
            Self::MissingColumn(column) => write!(formatter, "missing required column `{column}`"),
            Self::InvalidColumnType(column) => {
                write!(formatter, "column `{column}` has an unsupported type")
            }
            Self::NullValue { column, row } => {
                write!(formatter, "column `{column}` is null at row {row}")
            }
            Self::InvalidTimestamp { row, value } => {
                write!(formatter, "date has invalid timestamp {value} at row {row}")
            }
            Self::InvalidPrice { column, row, value } => {
                write!(
                    formatter,
                    "column `{column}` has invalid price {value} at row {row}"
                )
            }
            Self::InvalidValue { column, row, value } => {
                write!(formatter, "{column} must be finite at row {row}: {value}")
            }
            Self::InvalidHistoryValue { row } => write!(
                formatter,
                "open, close and open_interest must be positive/finite at row {row}"
            ),
            Self::EmptyTail => formatter.write_str("history tail must contain at least one bar"),
            Self::NonIncreasingTimestamp { row } => {
                write!(
                    formatter,
                    "date must strictly increase; row {row} is out of order"
                )
            }
        }
    }
}

impl Error for DataError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Open(error) => Some(error),
            Self::Parquet(error) => Some(error),
            Self::Arrow(error) => Some(error),
            Self::MissingColumn(_)
            | Self::InvalidColumnType(_)
            | Self::NullValue { .. }
            | Self::InvalidTimestamp { .. }
            | Self::InvalidPrice { .. }
            | Self::InvalidValue { .. }
            | Self::InvalidHistoryValue { .. }
            | Self::EmptyTail
            | Self::NonIncreasingTimestamp { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, RecordBatch, TimestampNanosecondArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    use super::read_bars;

    #[test]
    fn reads_date_timestamp_ohlc_columns_from_parquet() {
        let path = std::env::temp_dir().join(format!(
            "bullet-data-{}-{}.parquet",
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
            Field::new("open_interest", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(TimestampNanosecondArray::from(vec![1_i64, 2])) as ArrayRef,
                Arc::new(Float64Array::from(vec![100.0, 101.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![101.0, 102.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![10.0, 11.0])) as ArrayRef,
            ],
        )
        .expect("test batch has a valid schema");
        let file = File::create(&path).expect("temporary parquet path is writable");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer initializes");
        writer.write(&batch).expect("writer accepts batch");
        writer.close().expect("writer writes footer");

        let bars = read_bars(&path).expect("valid parquet bars are read");
        std::fs::remove_file(&path).expect("temporary parquet file is removed");

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].timestamp_ns, 1);
        assert_eq!(bars[1].open.value(), 101.0);
        assert_eq!(bars[1].close.value(), 102.0);
    }

    #[test]
    fn reads_tail_with_open_interest_for_live_splice() {
        let path = std::env::temp_dir().join(format!(
            "bullet-live-history-{}-{}.parquet",
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
            Field::new("open_interest", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(TimestampNanosecondArray::from(vec![1_i64, 2, 3])) as ArrayRef,
                Arc::new(Float64Array::from(vec![100.0, 101.0, 102.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![101.0, 102.0, 103.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![10.0, 11.0, 12.0])) as ArrayRef,
            ],
        )
        .expect("test batch has a valid schema");
        let file = File::create(&path).expect("temporary parquet path is writable");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer initializes");
        writer.write(&batch).expect("writer accepts batch");
        writer.close().expect("writer writes footer");

        let bars = super::read_history_tail(&path, 2).expect("history tail is read");
        std::fs::remove_file(&path).expect("temporary parquet file is removed");

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].timestamp_ns, 2);
        assert_eq!(bars[1].open_interest, 12.0);
    }
}

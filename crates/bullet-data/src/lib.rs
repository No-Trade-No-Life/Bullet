//! Parquet market-bar input for Bullet backtests.

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::Path;

use arrow_array::{Array, Float64Array, RecordBatch, UInt64Array};
use arrow_schema::ArrowError;
use bullet_core::{Bar, Price};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const TIMESTAMP_COLUMN: &str = "timestamp_ns";
const OPEN_COLUMN: &str = "open";
const CLOSE_COLUMN: &str = "close";

pub fn read_bars(path: impl AsRef<Path>) -> Result<Vec<Bar>, DataError> {
    let file = File::open(path).map_err(DataError::Open)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(DataError::Parquet)?
        .build()
        .map_err(DataError::Parquet)?;
    let mut bars = Vec::new();

    for batch in reader {
        let batch = batch.map_err(DataError::Arrow)?;
        append_batch(&mut bars, &batch)?;
    }

    validate_timestamps(&bars)?;
    Ok(bars)
}

fn append_batch(bars: &mut Vec<Bar>, batch: &RecordBatch) -> Result<(), DataError> {
    let timestamps = required_u64_column(batch, TIMESTAMP_COLUMN)?;
    let opens = required_f64_column(batch, OPEN_COLUMN)?;
    let closes = required_f64_column(batch, CLOSE_COLUMN)?;

    for row in 0..batch.num_rows() {
        let timestamp_ns = required_value(timestamps, row, TIMESTAMP_COLUMN)?;
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

fn required_u64_column<'a>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a UInt64Array, DataError> {
    batch
        .column_by_name(name)
        .ok_or(DataError::MissingColumn(name))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or(DataError::InvalidColumnType(name))
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

impl ArrayValue<u64> for UInt64Array {
    fn value(&self, row: usize) -> u64 {
        UInt64Array::value(self, row)
    }
}

impl ArrayValue<f64> for Float64Array {
    fn value(&self, row: usize) -> f64 {
        Float64Array::value(self, row)
    }
}

fn price(value: f64, row: usize, column: &'static str) -> Result<Price, DataError> {
    Price::new(value).ok_or(DataError::InvalidPrice { column, row, value })
}

fn validate_timestamps(bars: &[Bar]) -> Result<(), DataError> {
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
    InvalidPrice {
        column: &'static str,
        row: usize,
        value: f64,
    },
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
            Self::InvalidPrice { column, row, value } => {
                write!(
                    formatter,
                    "column `{column}` has invalid price {value} at row {row}"
                )
            }
            Self::NonIncreasingTimestamp { row } => {
                write!(
                    formatter,
                    "timestamp_ns must strictly increase; row {row} is out of order"
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
            | Self::InvalidPrice { .. }
            | Self::NonIncreasingTimestamp { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, RecordBatch, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    use super::read_bars;

    #[test]
    fn reads_required_bar_columns_from_parquet() {
        let path =
            std::env::temp_dir().join(format!("bullet-data-{}-{}.parquet", std::process::id(), 1));
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp_ns", DataType::UInt64, false),
            Field::new("open", DataType::Float64, false),
            Field::new("close", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![1_u64, 2])) as ArrayRef,
                Arc::new(Float64Array::from(vec![100.0, 101.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![101.0, 102.0])) as ArrayRef,
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
}

use crate::{EvaluationError, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{self, Write};

#[derive(Default)]
struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn hash<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    let mut writer = HashWriter::default();
    serde_json::to_writer(&mut writer, value)
        .map_err(|error| EvaluationError(format!("canonical serialization: {error}")))?;
    Ok(format!("{:x}", writer.0.finalize()))
}

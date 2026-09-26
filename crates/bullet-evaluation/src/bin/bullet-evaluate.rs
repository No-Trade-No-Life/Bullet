use bullet_evaluation::{EvaluationInput, evaluate};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 || args[0] == args[1] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: bullet-evaluate INPUT.json NEW_OUTPUT.json (output must not exist)",
        )
        .into());
    }
    let input: EvaluationInput = serde_json::from_reader(BufReader::new(File::open(&args[0])?))?;
    let result = evaluate(&input)?;
    let mut writer = BufWriter::new(File::create_new(&args[1])?);
    serde_json::to_writer(&mut writer, &result)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

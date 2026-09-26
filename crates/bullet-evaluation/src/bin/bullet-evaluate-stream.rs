use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::PathBuf;

use bullet_evaluation::{
    EvaluationConfig, JsonlReplayOptions, StreamStatus, evaluate_jsonl, hash_decision_jsonl,
    hash_market_jsonl,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let config_path = args.next().ok_or_else(usage)?;
    let market_path = args.next().ok_or_else(usage)?;
    let decision_path = args.next().ok_or_else(usage)?;
    let output_dir = args.next().ok_or_else(usage)?;
    let mut checkpoint_every = 100_000_usize;
    let mut stop_after = None;
    let mut resume = false;
    while let Some(argument) = args.next() {
        match argument.to_string_lossy().as_ref() {
            "--resume" => resume = true,
            "--checkpoint-every" => {
                checkpoint_every = args
                    .next()
                    .ok_or_else(usage)?
                    .to_string_lossy()
                    .parse()
                    .map_err(|_| usage())?;
            }
            "--stop-after" => {
                stop_after = Some(
                    args.next()
                        .ok_or_else(usage)?
                        .to_string_lossy()
                        .parse()
                        .map_err(|_| usage())?,
                );
            }
            _ => return Err(usage().into()),
        }
    }
    let config: EvaluationConfig =
        serde_json::from_reader(BufReader::new(File::open(&config_path)?))?;
    let market_path = PathBuf::from(market_path);
    let decision_path = PathBuf::from(decision_path);
    let output_dir = PathBuf::from(output_dir);
    let summary = evaluate_jsonl(JsonlReplayOptions {
        config,
        market_sha256: hash_market_jsonl(&market_path)?,
        decision_sha256: hash_decision_jsonl(&decision_path)?,
        market_path,
        decisions_path: decision_path,
        output_dir: output_dir.clone(),
        checkpoint_every_intervals: checkpoint_every,
        stop_after_intervals: stop_after,
        resume,
    })?;
    if matches!(summary.status, StreamStatus::Complete) {
        let path = output_dir.join("summary.json");
        let mut writer = BufWriter::new(File::create_new(&path)?);
        serde_json::to_writer_pretty(&mut writer, &summary)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn usage() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "usage: bullet-evaluate-stream CONFIG.json MARKET.jsonl DECISIONS.jsonl OUTPUT_DIR [--resume] [--checkpoint-every N] [--stop-after N]",
    )
}

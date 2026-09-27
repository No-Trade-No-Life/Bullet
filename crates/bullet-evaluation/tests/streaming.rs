use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use bullet_evaluation::{
    Accounting, CausalAvailability, EvaluationConfig, EvaluationInput, EventTime,
    JsonlReplayOptions, MarketPoint, StreamStatus, TargetDecision, TerminalPolicy, evaluate,
    evaluate_jsonl, hash_decision_jsonl, hash_market_jsonl,
};

fn time(value: u64) -> EventTime {
    EventTime {
        timestamp_ns: value,
        sequence: 0,
    }
}

fn config() -> EvaluationConfig {
    EvaluationConfig {
        accounting: Accounting::NormalizedExposureV1,
        instrument: "BTC".to_owned(),
        one_way_cost_bps: 10.0,
        slippage_bps: 0.0,
        terminal_policy: TerminalPolicy::Liquidate,
        sharpe_periods_per_year: 1.0,
        sharpe_standard_deviation_ddof: 1,
        annualization_days_per_year: 1.0,
        evaluation_days: vec![0],
    }
}

fn decision(time_value: u64, target: i64, hash: &str) -> TargetDecision {
    TargetDecision {
        decision_time: time(time_value - 1),
        execution_time: time(time_value),
        instrument: "BTC".to_owned(),
        target_units: target,
        model_vintage: Some("stream-test".to_owned()),
        signal_id: Some("stream".to_owned()),
        source_id: Some("fixture".to_owned()),
        state_code: Some("ready".to_owned()),
        diagnostic_json: serde_json::json!({"b": 2, "a": 1}),
        causal_availability: CausalAvailability {
            source_end: time(time_value - 2),
            available_at: time(time_value - 1),
        },
        provenance_hash: hash.to_owned(),
    }
}

fn input() -> EvaluationInput {
    EvaluationInput {
        schema_version: 1,
        config: config(),
        market: vec![
            MarketPoint {
                time: time(100),
                instrument: "BTC".to_owned(),
                price: 100.0,
            },
            MarketPoint {
                time: time(101),
                instrument: "BTC".to_owned(),
                price: 100.0,
            },
            MarketPoint {
                time: time(102),
                instrument: "BTC".to_owned(),
                price: 110.0,
            },
            MarketPoint {
                time: time(103),
                instrument: "BTC".to_owned(),
                price: 100.0,
            },
        ],
        decisions: vec![
            decision(
                101,
                1,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            decision(
                102,
                -1,
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
        ],
    }
}

fn write_jsonl<T: serde::Serialize>(path: &Path, values: &[T]) {
    let mut file = File::create(path).expect("create JSONL");
    for value in values {
        serde_json::to_writer(&mut file, value).expect("serialize JSONL");
        file.write_all(b"\n").expect("write newline");
    }
}

fn setup(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    fs::create_dir_all(root).expect("create stream test root");
    let config_path = root.join("config.json");
    let market_path = root.join("market.jsonl");
    let decisions_path = root.join("decisions.jsonl");
    let value = input();
    serde_json::to_writer(
        File::create(&config_path).expect("create config"),
        &value.config,
    )
    .expect("serialize config");
    write_jsonl(&market_path, &value.market);
    write_jsonl(&decisions_path, &value.decisions);
    (config_path, market_path, decisions_path)
}

fn options(
    config: EvaluationConfig,
    market: &Path,
    decisions: &Path,
    output: &Path,
) -> JsonlReplayOptions {
    JsonlReplayOptions {
        config,
        market_path: market.to_owned(),
        decisions_path: decisions.to_owned(),
        output_dir: output.to_owned(),
        market_sha256: hash_market_jsonl(market).expect("market hash"),
        decision_sha256: hash_decision_jsonl(decisions).expect("decision hash"),
        checkpoint_every_intervals: 1,
        stop_after_intervals: None,
        resume: false,
    }
}

#[test]
fn streaming_output_matches_batch_and_checkpoint_restart_is_byte_exact() {
    let root = std::env::temp_dir().join(format!("bullet-stream-test-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root).expect("clean old stream test");
    }
    let (config_path, market_path, decisions_path) = setup(&root);
    let value = input();
    let batch = evaluate(&value).expect("batch evaluation");

    let full_dir = root.join("full");
    let full = evaluate_jsonl(options(
        value.config.clone(),
        &market_path,
        &decisions_path,
        &full_dir,
    ))
    .expect("full streaming evaluation");
    assert!(matches!(full.status, StreamStatus::Complete));
    assert_eq!(
        full.audit.as_ref().expect("full audit").run_sha256,
        batch.audit.run_sha256
    );
    assert_eq!(
        full.metrics.expect("full metrics").total_return_units,
        batch.metrics.total_return_units
    );

    let resumed_dir = root.join("resumed");
    let mut stopped_options = options(
        value.config.clone(),
        &market_path,
        &decisions_path,
        &resumed_dir,
    );
    stopped_options.stop_after_intervals = Some(2);
    let stopped = evaluate_jsonl(stopped_options).expect("checkpointed prefix");
    assert!(matches!(stopped.status, StreamStatus::CheckpointOnly));
    assert_eq!(stopped.processed_intervals, 2);

    let resumed = evaluate_jsonl(JsonlReplayOptions {
        config: value.config,
        market_path: market_path.clone(),
        decisions_path: decisions_path.clone(),
        output_dir: resumed_dir.clone(),
        market_sha256: hash_market_jsonl(&market_path).expect("market hash"),
        decision_sha256: hash_decision_jsonl(&decisions_path).expect("decision hash"),
        checkpoint_every_intervals: 1,
        stop_after_intervals: None,
        resume: true,
    })
    .expect("resume streaming evaluation");
    assert!(matches!(resumed.status, StreamStatus::Complete));
    assert_eq!(
        resumed.audit.expect("resumed audit").run_sha256,
        full.audit.expect("full audit").run_sha256
    );
    for name in [
        "decision_ledger.jsonl",
        "execution_ledger.jsonl",
        "position_ledger.jsonl",
        "daily_returns.jsonl",
    ] {
        assert_eq!(
            fs::read(full_dir.join(name)).expect("full output"),
            fs::read(resumed_dir.join(name)).expect("resumed output"),
            "output differs for {name}"
        );
    }
    let _ = fs::remove_file(config_path);
    fs::remove_dir_all(root).expect("clean stream test");
}

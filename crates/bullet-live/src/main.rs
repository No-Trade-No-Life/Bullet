mod config;
mod ctpd;
mod linkit;
mod market_time;
mod model;
mod parity;
mod protocol;

use std::{
    error::Error,
    fs::OpenOptions,
    io::{BufWriter, Write},
    sync::{Arc, RwLock},
    time::Instant,
};

use config::{InstrumentConfig, LiveConfig};
use model::{CtpdTick, LiveTradeSignal, Portfolio};
use protocol::RemoteAccountState;
use tokio::sync::{Mutex, mpsc};

const LINKIT_SIGNAL_QUEUE_CAPACITY: usize = 64;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("bullet-live: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    match arguments.next().as_deref() {
        Some("serve") => {
            let path = arguments
                .next()
                .ok_or("usage: bullet-live serve <config.toml>")?;
            if arguments.next().is_some() {
                return Err("usage: bullet-live serve <config.toml>".into());
            }
            serve(&path).await
        }
        Some("benchmark") => {
            let events = arguments
                .next()
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(20_000);
            if arguments.next().is_some() || events == 0 {
                return Err("usage: bullet-live benchmark [event_count]".into());
            }
            benchmark(events)
        }
        Some("seed-benchmark") => {
            let path = arguments
                .next()
                .ok_or("usage: bullet-live seed-benchmark <config.toml>")?;
            if arguments.next().is_some() {
                return Err("usage: bullet-live seed-benchmark <config.toml>".into());
            }
            seed_benchmark(&path)
        }
        Some("replay") => {
            let config = arguments
                .next()
                .ok_or("usage: bullet-live replay <config.toml> <output.jsonl>")?;
            let output = arguments
                .next()
                .ok_or("usage: bullet-live replay <config.toml> <output.jsonl>")?;
            if arguments.next().is_some() {
                return Err("usage: bullet-live replay <config.toml> <output.jsonl>".into());
            }
            replay(&config, &output)
        }
        Some("verify-parity") => {
            let config = arguments.next().ok_or(
                "usage: bullet-live verify-parity <config.toml> <lab_candidate_decisions.csv> <lab_raw_candidate_labels.csv>",
            )?;
            let decisions = arguments.next().ok_or(
                "usage: bullet-live verify-parity <config.toml> <lab_candidate_decisions.csv> <lab_raw_candidate_labels.csv>",
            )?;
            let labels = arguments.next().ok_or(
                "usage: bullet-live verify-parity <config.toml> <lab_candidate_decisions.csv> <lab_raw_candidate_labels.csv>",
            )?;
            if arguments.next().is_some() {
                return Err("usage: bullet-live verify-parity <config.toml> <lab_candidate_decisions.csv> <lab_raw_candidate_labels.csv>".into());
            }
            verify_parity(&config, &decisions, &labels)
        }
        _ => Err(
            "usage: bullet-live <serve <config.toml>|benchmark [event_count]|seed-benchmark <config.toml>|replay <config.toml> <output.jsonl>|verify-parity <config.toml> <lab_candidate_decisions.csv> <lab_raw_candidate_labels.csv>>"
                .into(),
        ),
    }
}

async fn serve(path: &str) -> Result<(), Box<dyn Error>> {
    let (config, secrets) = LiveConfig::load(path)?;
    let portfolio = Arc::new(RwLock::new(seed_portfolio(&config, false)?));
    let client = reqwest::Client::builder().build()?;
    let instruments = Arc::new(config.instruments.clone());
    ctpd::recover_and_synchronize(
        &client,
        &config.ctpd,
        &secrets.ctpd_bearer_token,
        &instruments,
        &portfolio,
    )
    .await?;
    let signal_tx = start_linkit_sender(&client, &config, secrets.linkit_bearer_token).await?;
    let recovery_gate = Arc::new(Mutex::new(()));
    for instrument in instruments.iter().cloned() {
        tokio::spawn(ctpd::consume_ticks(
            client.clone(),
            config.ctpd.clone(),
            secrets.ctpd_bearer_token.clone(),
            instrument,
            ctpd::LiveState {
                instruments: instruments.clone(),
                portfolio: portfolio.clone(),
                recovery_gate: recovery_gate.clone(),
            },
            signal_tx.clone(),
        ));
    }
    let app = protocol::app(RemoteAccountState {
        account_id: config.account_id,
        bearer_token: secrets.remote_bearer_token,
        portfolio,
    });
    let listener = tokio::net::TcpListener::bind(config.bind_address).await?;
    println!(
        "bullet-live lab0334 remote-account listener: {}",
        config.bind_address
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn start_linkit_sender(
    client: &reqwest::Client,
    config: &LiveConfig,
    bearer_token: Option<String>,
) -> Result<Option<mpsc::Sender<LiveTradeSignal>>, Box<dyn Error>> {
    let Some(linkit) = config.linkit.clone() else {
        return Ok(None);
    };
    let bearer_token = bearer_token.expect("Linkit config loads its token file");
    linkit::validate_group(client, &linkit, &bearer_token).await?;
    let (sender, receiver) = mpsc::channel(LINKIT_SIGNAL_QUEUE_CAPACITY);
    tokio::spawn(linkit::send_loop(
        client.clone(),
        linkit,
        bearer_token,
        receiver,
    ));
    Ok(Some(sender))
}

fn seed_portfolio(config: &LiveConfig, capture_ledger: bool) -> Result<Portfolio, String> {
    let mut portfolio = Portfolio::default();
    portfolio.set_capture_ledger(capture_ledger);
    let mut histories = Vec::new();
    for instrument in &config.instruments {
        let history = bullet_data::read_history_tail(&instrument.parquet, config.history_seed_bars)
            .map_err(|error| format!("cannot read {}: {error}", instrument.symbol))?;
        if history.len() < 5_000 {
            return Err(format!(
                "{} has fewer than 5000 Parquet bars",
                instrument.symbol
            ));
        }
        portfolio.insert(
            instrument.market_instrument_id.clone(),
            Portfolio::new_model(instrument),
        );
        portfolio.set_history_day_bar_counts(&instrument.market_instrument_id, &history)?;
        histories.push((instrument.market_instrument_id.clone(), history, 0_usize));
    }
    // Four-way merge preserves the lab's `entry_time, trade_type, symbol`
    // causal ordering without copying the 102 MiB Parquet corpus again.
    loop {
        let next = histories
            .iter()
            .enumerate()
            .filter_map(|(index, (market, bars, cursor))| {
                bars.get(*cursor)
                    .map(|bar| (index, market.as_str(), bar.timestamp_ns))
            })
            .min_by(|left, right| left.2.cmp(&right.2).then_with(|| left.1.cmp(right.1)))
            .map(|(index, _, _)| index);
        let Some(index) = next else { break };
        let (market, bar) = {
            let (market, bars, cursor) = &mut histories[index];
            let bar = bars[*cursor].clone();
            *cursor += 1;
            (market.clone(), bar)
        };
        portfolio.ingest_history(&market, &bar)?;
        let next_timestamp = histories
            .iter()
            .filter_map(|(_, bars, cursor)| bars.get(*cursor).map(|next| next.timestamp_ns))
            .min();
        if next_timestamp != Some(bar.timestamp_ns) {
            portfolio.flush_history_candidates();
        }
    }
    Ok(portfolio)
}

fn benchmark(events: usize) -> Result<(), Box<dyn Error>> {
    let config = InstrumentConfig {
        symbol: "IF8888".into(),
        market_instrument_id: "IDX-CFFEX-IF".into(),
        target_instrument_id: "IF2609".into(),
        parquet: "/not-used-by-benchmark".into(),
        full_weight_contracts: 10.0,
        contract_multiplier: 300.0,
        session_bar_count: 240,
        session_end_time: "15:00:00".into(),
    };
    let mut initial = Portfolio::default();
    initial.insert(
        config.market_instrument_id.clone(),
        Portfolio::new_model(&config),
    );
    initial.install_benchmark_target();
    let portfolio = RwLock::new(initial);
    let mut samples = Vec::with_capacity(events);
    for event in 0..events {
        let started = Instant::now();
        portfolio
            .write()
            .expect("benchmark portfolio lock poisoned")
            .ingest(benchmark_tick(event))?;
        samples.push(started.elapsed().as_nanos());
    }
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p99 = samples[samples.len() * 99 / 100];
    let maximum = *samples.last().expect("events is non-empty");
    println!(
        "inference_events={events} p50_ns={p50} p99_ns={p99} max_ns={maximum} budget_ns=100000000"
    );
    if p99 >= 100_000_000 {
        return Err(format!("p99 inference latency exceeds 100ms: {p99}ns").into());
    }
    Ok(())
}

fn seed_benchmark(path: &str) -> Result<(), Box<dyn Error>> {
    let config = LiveConfig::load_without_secrets(path)?;
    let started = Instant::now();
    let portfolio = seed_portfolio(&config, false)?;
    let elapsed = started.elapsed();
    println!(
        "history_seed_bars={} seeded_models={} elapsed_ms={}",
        config.history_seed_bars,
        portfolio.model_count(),
        elapsed.as_millis(),
    );
    Ok(())
}

/// Rebuilds the causal in-memory state from Parquet and writes the live
/// arbitrator's decision ledger.  This is an offline verification command;
/// it deliberately shares `seed_portfolio` with production instead of
/// maintaining a separate replay model.
fn replay(config_path: &str, output_path: &str) -> Result<(), Box<dyn Error>> {
    let config = LiveConfig::load_without_secrets(config_path)?;
    let portfolio = seed_portfolio(&config, true)?;
    let labels_path = format!("{output_path}.labels.jsonl");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)?;
    let labels_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&labels_path)?;
    let mut writer = BufWriter::new(file);
    let mut labels_writer = BufWriter::new(labels_file);
    for decision in portfolio.candidate_decisions() {
        serde_json::to_writer(&mut writer, decision)?;
        writer.write_all(b"\n")?;
    }
    for label in portfolio.candidate_labels() {
        serde_json::to_writer(&mut labels_writer, label)?;
        labels_writer.write_all(b"\n")?;
    }
    writer.flush()?;
    labels_writer.flush()?;
    println!(
        "replay_candidates={} replay_labels={} output={output_path}",
        portfolio.candidate_decisions().len(),
        portfolio.candidate_labels().len()
    );
    Ok(())
}

fn verify_parity(
    config_path: &str,
    reference_decisions_path: &str,
    reference_labels_path: &str,
) -> Result<(), Box<dyn Error>> {
    let config = LiveConfig::load_without_secrets(config_path)?;
    let portfolio = seed_portfolio(&config, true)?;
    let summary = parity::verify(&portfolio, reference_decisions_path, reference_labels_path)?;
    println!(
        "parity=pass decisions={} labels={} canonical_bytes={}",
        summary.decisions, summary.labels, summary.canonical_bytes
    );
    Ok(())
}

fn benchmark_tick(event: usize) -> CtpdTick {
    let start = chrono::NaiveDateTime::parse_from_str("20260810 09:30:00", "%Y%m%d %H:%M:%S")
        .expect("benchmark start is valid");
    let at = start + chrono::Duration::minutes(event as i64);
    CtpdTick {
        instrument_id: "IDX-CFFEX-IF".into(),
        exchange_id: "CFFEX".into(),
        trading_day: at.format("%Y%m%d").to_string(),
        action_day: at.format("%Y%m%d").to_string(),
        update_time: at.format("%H:%M:%S").to_string(),
        update_millisec: 0,
        last_price: 4_000.0 + (event % 5) as f64,
        volume: event as f64 + 1.0,
        turnover: 4_000.0 * (event + 1) as f64,
        open_interest: event as f64 + 1.0,
    }
}

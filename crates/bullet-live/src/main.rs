mod config;
mod ctpd;
mod model;
mod protocol;

use std::{
    error::Error,
    sync::{Arc, RwLock},
    time::Instant,
};

use config::{InstrumentConfig, LiveConfig};
use model::{CtpdTick, Portfolio};
use protocol::RemoteAccountState;

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
        _ => Err(
            "usage: bullet-live <serve <config.toml>|benchmark [event_count]|seed-benchmark <config.toml>>"
                .into(),
        ),
    }
}

async fn serve(path: &str) -> Result<(), Box<dyn Error>> {
    let (config, secrets) = LiveConfig::load(path)?;
    let portfolio = Arc::new(RwLock::new(seed_portfolio(&config)?));
    let client = reqwest::Client::builder().build()?;
    for instrument in config.instruments.clone() {
        tokio::spawn(ctpd::consume_ticks(
            client.clone(),
            config.ctpd.clone(),
            secrets.ctpd_bearer_token.clone(),
            instrument,
            portfolio.clone(),
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

fn seed_portfolio(config: &LiveConfig) -> Result<Portfolio, String> {
    let mut portfolio = Portfolio::default();
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
    let portfolio = seed_portfolio(&config)?;
    let elapsed = started.elapsed();
    println!(
        "history_seed_bars={} seeded_models={} elapsed_ms={}",
        config.history_seed_bars,
        portfolio.model_count(),
        elapsed.as_millis(),
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

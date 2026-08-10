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
use model::{CtpdTick, Lab0344Model, Portfolio};
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
        _ => Err("usage: bullet-live <serve <config.toml>|benchmark [event_count]>".into()),
    }
}

async fn serve(path: &str) -> Result<(), Box<dyn Error>> {
    let (config, secrets) = LiveConfig::load(path)?;
    let portfolio = Arc::new(RwLock::new(seed_portfolio(&config)?));
    let client = reqwest::Client::builder().build()?;
    for instrument in config.instruments.clone() {
        let feed_client = client.clone();
        let feed_ctpd = config.ctpd.clone();
        let feed_token = secrets.ctpd_bearer_token.clone();
        let feed_portfolio = portfolio.clone();
        tokio::spawn(async move {
            ctpd::consume_ticks(
                feed_client,
                feed_ctpd,
                feed_token,
                instrument,
                feed_portfolio,
            )
            .await;
        });
    }
    let app = protocol::app(RemoteAccountState {
        account_id: config.account_id,
        bearer_token: secrets.remote_bearer_token,
        portfolio,
    });
    let listener = tokio::net::TcpListener::bind(config.bind_address).await?;
    println!(
        "bullet-live remote-account listener: {}",
        config.bind_address
    );
    axum::serve(listener, app).await?;
    Ok(())
}

fn seed_portfolio(config: &LiveConfig) -> Result<Portfolio, String> {
    let mut portfolio = Portfolio::default();
    for instrument in &config.instruments {
        let history = bullet_data::read_history_tail(&instrument.parquet, config.history_tail_bars)
            .map_err(|error| format!("cannot seed {}: {error}", instrument.symbol))?;
        let mut model = Lab0344Model::new(instrument);
        model.seed_history(&history)?;
        portfolio.insert(instrument.ctpd_instrument_id.clone(), model);
    }
    Ok(portfolio)
}

fn benchmark(events: usize) -> Result<(), Box<dyn Error>> {
    let config = InstrumentConfig {
        symbol: "IF".into(),
        ctpd_instrument_id: "IF2609".into(),
        parquet: "/not-used-by-benchmark".into(),
        target_contracts: 1,
        contract_multiplier: 300.0,
        session_bar_count: events.max(80),
        last_executable_signal_time: "14:40:00".into(),
    };
    let mut initial = Portfolio::default();
    initial.insert("IF2609".into(), Lab0344Model::new(&config));
    let portfolio = RwLock::new(initial);
    let mut samples = Vec::with_capacity(events);
    for event in 0..events {
        let tick = benchmark_tick(event);
        let started = Instant::now();
        portfolio
            .write()
            .expect("benchmark portfolio lock poisoned")
            .ingest(tick)?;
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

fn benchmark_tick(event: usize) -> CtpdTick {
    let start = chrono::NaiveDateTime::parse_from_str("20260810 09:30:00", "%Y%m%d %H:%M:%S")
        .expect("benchmark start is valid");
    let at = start + chrono::Duration::minutes(event as i64);
    CtpdTick {
        instrument_id: "IF2609".into(),
        exchange_id: "CFFEX".into(),
        trading_day: at.format("%Y%m%d").to_string(),
        action_day: at.format("%Y%m%d").to_string(),
        update_time: at.format("%H:%M:%S").to_string(),
        update_millisec: 0,
        last_price: 4_000.0 + (event % 5) as f64,
        open_interest: event as f64 + 1.0,
    }
}

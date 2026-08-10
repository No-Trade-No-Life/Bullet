use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use bullet_data::HistoryBar;
use chrono::{DateTime, FixedOffset, TimeZone, Utc};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use tokio::time::{sleep, timeout};

use crate::{
    config::{CtpdConfig, InstrumentConfig},
    model::{CtpdTick, Portfolio},
};

const KLINE_RECOVERY_CHUNK_MS: i64 = 24 * 60 * 60 * 1_000;
const KLINE_RECOVERY_TIMEOUT: Duration = Duration::from_secs(30);

/// Runs one reconnecting CTPD SSE consumer. A stale or disconnected stream
/// clears its target before retrying, so 1Exchange never sees a target backed
/// by an unknown market state.
pub async fn consume_ticks(
    client: Client,
    ctpd: CtpdConfig,
    bearer_token: String,
    instrument: InstrumentConfig,
    portfolio: Arc<RwLock<Portfolio>>,
) {
    loop {
        let result =
            consume_connection(&client, &ctpd, &bearer_token, &instrument, &portfolio).await;
        if let Err(error) = result {
            eprintln!(
                "bullet-live: CTPD feed {}: {error}",
                instrument.market_instrument_id
            );
        }
        portfolio
            .write()
            .expect("portfolio lock poisoned")
            .clear_all();
        sleep(Duration::from_millis(250)).await;
    }
}

async fn consume_connection(
    client: &Client,
    ctpd: &CtpdConfig,
    bearer_token: &str,
    instrument: &InstrumentConfig,
    portfolio: &Arc<RwLock<Portfolio>>,
) -> Result<(), String> {
    recover_completed_klines(client, ctpd, bearer_token, instrument, portfolio).await?;
    portfolio
        .write()
        .expect("portfolio lock poisoned")
        .mark_market_synchronized(&instrument.market_instrument_id)?;
    let url = format!(
        "{}/v1/ticks?instrument_id={}",
        ctpd.base_url.trim_end_matches('/'),
        instrument.market_instrument_id
    );
    let response = client
        .get(url)
        .bearer_auth(bearer_token)
        .send()
        .await
        .map_err(|error| format!("request failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("CTPD rejected subscription: {error}"))?;
    let stale_after = Duration::from_millis(ctpd.stale_after_ms);
    let mut stream = response.bytes_stream();
    let mut decoder = SseDecoder::default();

    loop {
        let next = timeout(stale_after, stream.next())
            .await
            .map_err(|_| format!("no CTPD tick for {} ms", ctpd.stale_after_ms))?;
        let bytes = next
            .ok_or("CTPD SSE stream ended")?
            .map_err(|error| format!("CTPD SSE read failed: {error}"))?;
        for payload in decoder.push(&bytes)? {
            let tick: CtpdTick = serde_json::from_str(&payload)
                .map_err(|error| format!("invalid CTPD tick: {error}"))?;
            portfolio
                .write()
                .expect("portfolio lock poisoned")
                .ingest(tick)?;
        }
    }
}

#[derive(Debug, Deserialize)]
struct CtpdKline {
    start_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: i64,
    money: f64,
    open_interest: f64,
    closed: bool,
}

async fn recover_completed_klines(
    client: &Client,
    ctpd: &CtpdConfig,
    bearer_token: &str,
    instrument: &InstrumentConfig,
    portfolio: &Arc<RwLock<Portfolio>>,
) -> Result<(), String> {
    let last_market_ns = portfolio
        .read()
        .expect("portfolio lock poisoned")
        .last_bar_timestamp_ns(&instrument.market_instrument_id)
        .ok_or("cannot recover CTPD Klines before Parquet history is seeded")?;
    let last_ms = ctpd_ms_from_market_ns(last_market_ns)?;
    let now_ms = chrono::Utc::now().timestamp_millis();
    if now_ms < last_ms {
        return Err("CTPD clock precedes the seeded Parquet history".into());
    }
    let mut recovered = 0_usize;
    for (start_ms, end_ms) in kline_recovery_windows(last_ms, now_ms) {
        let url = format!(
            "{}/v1/klines?instrument_id={}&start_ms={}&end_ms={}&interval_secs=60",
            ctpd.base_url.trim_end_matches('/'),
            instrument.market_instrument_id,
            start_ms,
            end_ms,
        );
        let bytes = client
            .get(url)
            .bearer_auth(bearer_token)
            .timeout(KLINE_RECOVERY_TIMEOUT)
            .send()
            .await
            .map_err(|error| format!("Kline recovery request failed: {error}"))?
            .error_for_status()
            .map_err(|error| format!("CTPD rejected Kline recovery: {error}"))?
            .bytes()
            .await
            .map_err(|error| format!("cannot read CTPD Kline response: {error}"))?;
        let mut klines = serde_json::from_slice::<Vec<CtpdKline>>(&bytes)
            .map_err(|error| format!("invalid CTPD Kline response: {error}"))?;
        klines.sort_by_key(|kline| kline.start_ms);
        for kline in klines.into_iter().filter(|kline| kline.closed) {
            let bar = history_bar_from_kline(kline)?;
            if portfolio
                .write()
                .expect("portfolio lock poisoned")
                .ingest_recovered_history(&instrument.market_instrument_id, &bar)?
            {
                recovered += 1;
            }
        }
    }
    if recovered > 0 {
        eprintln!(
            "bullet-live: recovered {recovered} completed CTPD Klines for {}",
            instrument.market_instrument_id
        );
    }
    Ok(())
}

fn kline_recovery_windows(start_ms: i64, end_ms: i64) -> Vec<(i64, i64)> {
    let mut windows = Vec::new();
    let mut cursor = start_ms;
    while cursor < end_ms {
        let next = cursor.saturating_add(KLINE_RECOVERY_CHUNK_MS).min(end_ms);
        windows.push((cursor, next));
        cursor = next;
    }
    windows
}

fn history_bar_from_kline(kline: CtpdKline) -> Result<HistoryBar, String> {
    if kline.start_ms < 0
        || !kline.open.is_finite()
        || !kline.high.is_finite()
        || !kline.low.is_finite()
        || !kline.close.is_finite()
        || !kline.money.is_finite()
        || !kline.open_interest.is_finite()
        || kline.open <= 0.0
        || kline.high < kline.open
        || kline.high < kline.close
        || kline.low <= 0.0
        || kline.low > kline.open
        || kline.low > kline.close
        || kline.volume < 0
        || kline.money < 0.0
        || kline.open_interest < 0.0
    {
        return Err("CTPD Kline has invalid market values".into());
    }
    let timestamp_ns = market_ns_from_ctpd_ms(
        kline
            .start_ms
            .checked_add(60_000)
            .ok_or("CTPD Kline timestamp overflow")?,
    )?;
    Ok(HistoryBar {
        timestamp_ns,
        open: kline.open,
        high: kline.high,
        low: kline.low,
        close: kline.close,
        volume: kline.volume as f64,
        money: kline.money,
        open_interest: kline.open_interest,
    })
}

/// Parquet timestamps encode CFFEX's Shanghai wall-clock labels without a
/// timezone. CTPD's Kline query uses true Unix milliseconds. Keep the model
/// in the Parquet coordinate and perform this conversion only at the HTTP
/// boundary; treating one representation as the other shifts the strategy by
/// eight hours.
fn ctpd_ms_from_market_ns(timestamp_ns: u64) -> Result<i64, String> {
    let pseudo_utc = DateTime::<Utc>::from_timestamp_nanos(
        i64::try_from(timestamp_ns).map_err(|_| "market timestamp outside chrono range")?,
    );
    beijing()
        .from_local_datetime(&pseudo_utc.naive_utc())
        .single()
        .map(|datetime| datetime.with_timezone(&Utc).timestamp_millis())
        .ok_or("market timestamp is not an unambiguous Shanghai time".into())
}

fn market_ns_from_ctpd_ms(timestamp_ms: i64) -> Result<u64, String> {
    let market = DateTime::<Utc>::from_timestamp_millis(timestamp_ms)
        .ok_or("CTPD Kline timestamp outside chrono range")?
        .with_timezone(&beijing())
        .naive_local();
    u64::try_from(
        market
            .and_utc()
            .timestamp_nanos_opt()
            .ok_or("CTPD Kline timestamp outside chrono range")?,
    )
    .map_err(|_| "CTPD Kline timestamp before epoch".into())
}

fn beijing() -> FixedOffset {
    FixedOffset::east_opt(8 * 3_600).expect("China Standard Time offset is valid")
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=newline).collect::<Vec<_>>();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = std::str::from_utf8(&line).map_err(|_| "CTPD SSE is not UTF-8")?;
            if line.is_empty() {
                if self.event.as_deref() == Some("tick") && !self.data.is_empty() {
                    events.push(self.data.join("\n"));
                }
                self.event = None;
                self.data.clear();
                continue;
            }
            if let Some(value) = line.strip_prefix("event:") {
                self.event = Some(value.trim_start().to_owned());
            } else if let Some(value) = line.strip_prefix("data:") {
                self.data.push(value.trim_start().to_owned());
            }
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::IntoFuture,
        path::PathBuf,
        sync::{Arc, RwLock},
    };

    use axum::{
        Json, Router,
        extract::State,
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::get,
    };
    use chrono::{Duration, NaiveDateTime};
    use reqwest::Client;

    use super::{
        CtpdKline, SseDecoder, consume_connection, ctpd_ms_from_market_ns, history_bar_from_kline,
        kline_recovery_windows,
    };
    use crate::{
        config::{CtpdConfig, InstrumentConfig},
        model::Portfolio,
    };
    use bullet_data::HistoryBar;

    #[test]
    fn sse_decoder_handles_split_lines_and_ignores_other_events() {
        let mut decoder = SseDecoder::default();
        assert!(
            decoder
                .push(b"event: tick\ndata: {\"instrument_id\":")
                .unwrap()
                .is_empty()
        );
        let values = decoder
            .push(b"\"IF2609\"}\n\nevent: keepalive\ndata: x\n\n")
            .unwrap();
        assert_eq!(values, [r#"{"instrument_id":"IF2609"}"#]);
    }

    #[test]
    fn converts_ctpd_utc_klines_to_parquet_market_labels() {
        let start_ms = NaiveDateTime::parse_from_str("20260810 01:30:00", "%Y%m%d %H:%M:%S")
            .unwrap()
            .and_utc()
            .timestamp_millis();
        let bar = history_bar_from_kline(CtpdKline {
            start_ms,
            open: 4_000.0,
            high: 4_010.0,
            low: 3_990.0,
            close: 4_005.0,
            volume: 1,
            money: 4_005.0,
            open_interest: 1.0,
            closed: true,
        })
        .unwrap();
        let expected = NaiveDateTime::parse_from_str("20260810 09:31:00", "%Y%m%d %H:%M:%S")
            .unwrap()
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap() as u64;
        assert_eq!(bar.timestamp_ns, expected);
        assert_eq!(ctpd_ms_from_market_ns(expected).unwrap(), start_ms + 60_000);
    }

    #[test]
    fn splits_stale_parquet_recovery_into_bounded_daily_windows() {
        let day = 24 * 60 * 60 * 1_000;
        assert_eq!(
            kline_recovery_windows(1_000, 1_000 + 2 * day + 1),
            vec![
                (1_000, 1_000 + day),
                (1_000 + day, 1_000 + 2 * day),
                (1_000 + 2 * day, 1_000 + 2 * day + 1),
            ]
        );
        assert!(kline_recovery_windows(1_000, 1_000).is_empty());
    }

    #[tokio::test]
    async fn consumes_ctpd_sse_with_required_bar_fields() {
        let payload = (0..61)
            .map(|minute| {
                let at = NaiveDateTime::parse_from_str("20260810 09:30:00", "%Y%m%d %H:%M:%S")
                    .unwrap()
                    + Duration::minutes(minute);
                format!(
                    "event: tick\ndata: {{\"instrument_id\":\"IDX-CFFEX-IF\",\"exchange_id\":\"CFFEX\",\"trading_day\":\"{}\",\"action_day\":\"{}\",\"update_time\":\"{}\",\"update_millisec\":0,\"last_price\":{},\"volume\":{},\"turnover\":{},\"open_interest\":{}}}\n\n",
                    at.format("%Y%m%d"),
                    at.format("%Y%m%d"),
                    at.format("%H:%M:%S"),
                    4_000 + minute,
                    minute + 1,
                    (4_000 + minute) * (minute + 1),
                    minute + 1,
                )
            })
            .collect::<String>();
        let server = Router::new()
            .route("/v1/ticks", get(mock_ticks))
            .route("/v1/klines", get(mock_klines))
            .with_state(payload);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_task = tokio::spawn(axum::serve(listener, server).into_future());
        let instrument = InstrumentConfig {
            symbol: "IF8888".into(),
            market_instrument_id: "IDX-CFFEX-IF".into(),
            target_instrument_id: "IF2609".into(),
            parquet: "/unused".into(),
            full_weight_contracts: 10.0,
            contract_multiplier: 300.0,
            session_bar_count: 240,
            session_end_time: "15:00:00".into(),
        };
        let mut initial = Portfolio::default();
        initial.insert(
            instrument.market_instrument_id.clone(),
            Portfolio::new_model(&instrument),
        );
        initial
            .ingest_history(
                &instrument.market_instrument_id,
                &HistoryBar {
                    timestamp_ns: NaiveDateTime::parse_from_str(
                        "20260810 09:30:00",
                        "%Y%m%d %H:%M:%S",
                    )
                    .unwrap()
                    .and_utc()
                    .timestamp_nanos_opt()
                    .unwrap() as u64,
                    open: 4_000.0,
                    high: 4_000.0,
                    low: 4_000.0,
                    close: 4_000.0,
                    volume: 1.0,
                    money: 4_000.0,
                    open_interest: 1.0,
                },
            )
            .unwrap();
        let portfolio = Arc::new(RwLock::new(initial));
        let error = consume_connection(
            &Client::new(),
            &CtpdConfig {
                base_url: format!("http://{address}"),
                bearer_token_file: PathBuf::from("/unused"),
                stale_after_ms: 1_000,
            },
            "ctpd-test-token",
            &instrument,
            &portfolio,
        )
        .await
        .unwrap_err();
        server_task.abort();
        assert_eq!(error, "CTPD SSE stream ended");
        assert!(portfolio.read().unwrap().targets().is_empty());
    }

    async fn mock_ticks(State(payload): State<String>, headers: HeaderMap) -> impl IntoResponse {
        if headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            != Some("Bearer ctpd-test-token")
        {
            return (StatusCode::UNAUTHORIZED, String::new());
        }
        (StatusCode::OK, payload)
    }

    async fn mock_klines(headers: HeaderMap) -> impl IntoResponse {
        if headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            != Some("Bearer ctpd-test-token")
        {
            return (
                StatusCode::UNAUTHORIZED,
                Json(Vec::<serde_json::Value>::new()),
            );
        }
        (StatusCode::OK, Json(Vec::<serde_json::Value>::new()))
    }
}

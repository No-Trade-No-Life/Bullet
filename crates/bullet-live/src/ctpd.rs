use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use bullet_data::HistoryBar;
use chrono::{DateTime, Timelike, Utc};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use tokio::{
    sync::Mutex,
    time::{sleep, timeout},
};

use crate::{
    config::{CtpdConfig, InstrumentConfig},
    market_time::{ctpd_ms_from_timestamp_ns, shanghai, timestamp_ns_from_ctpd_ms},
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
    instruments: Arc<Vec<InstrumentConfig>>,
    portfolio: Arc<RwLock<Portfolio>>,
    recovery_gate: Arc<Mutex<()>>,
) {
    let mut recover_before_connect = false;
    let connection = CtpdConnection {
        client: &client,
        ctpd: &ctpd,
        bearer_token: &bearer_token,
        instruments: &instruments,
        portfolio: &portfolio,
        recovery_gate: &recovery_gate,
    };
    loop {
        let result = consume_connection(&connection, &instrument, recover_before_connect).await;
        let expected_lunch_break = is_cffex_lunch_break(Utc::now());
        if let Err(error) = result
            && !expected_lunch_break
        {
            eprintln!(
                "bullet-live: CTPD feed {}: {error}",
                instrument.market_instrument_id
            );
        }
        if !expected_lunch_break {
            let _recovery = recovery_gate.lock().await;
            portfolio
                .write()
                .expect("portfolio lock poisoned")
                .clear_all();
        }
        recover_before_connect = true;
        sleep(Duration::from_millis(250)).await;
    }
}

/// lab0334 can hold a virtual leg from the morning into the afternoon session.
/// CFFEX has no ticks from 11:30 through 12:59 China Standard Time, so that
/// expected silence cannot invalidate an already synchronized target.
fn is_cffex_lunch_break(now: DateTime<Utc>) -> bool {
    let local = now.with_timezone(&shanghai());
    (local.hour() == 11 && local.minute() >= 30) || local.hour() == 12
}

struct CtpdConnection<'a> {
    client: &'a Client,
    ctpd: &'a CtpdConfig,
    bearer_token: &'a str,
    instruments: &'a [InstrumentConfig],
    portfolio: &'a Arc<RwLock<Portfolio>>,
    recovery_gate: &'a Arc<Mutex<()>>,
}

async fn consume_connection(
    connection: &CtpdConnection<'_>,
    instrument: &InstrumentConfig,
    recover_before_connect: bool,
) -> Result<(), String> {
    if recover_before_connect {
        let _recovery = connection.recovery_gate.lock().await;
        recover_and_synchronize(
            connection.client,
            connection.ctpd,
            connection.bearer_token,
            connection.instruments,
            connection.portfolio,
        )
        .await?;
    }
    let url = format!(
        "{}/v1/ticks?instrument_id={}",
        connection.ctpd.base_url.trim_end_matches('/'),
        instrument.market_instrument_id
    );
    let response = connection
        .client
        .get(url)
        .bearer_auth(connection.bearer_token)
        .send()
        .await
        .map_err(|error| format!("request failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("CTPD rejected subscription: {error}"))?;
    let stale_after = Duration::from_millis(connection.ctpd.stale_after_ms);
    let mut stream = response.bytes_stream();
    let mut decoder = SseDecoder::default();

    loop {
        let next = timeout(stale_after, stream.next())
            .await
            .map_err(|_| format!("no CTPD tick for {} ms", connection.ctpd.stale_after_ms))?;
        let bytes = next
            .ok_or("CTPD SSE stream ended")?
            .map_err(|error| format!("CTPD SSE read failed: {error}"))?;
        for payload in decoder.push(&bytes)? {
            let tick: CtpdTick = serde_json::from_str(&payload)
                .map_err(|error| format!("invalid CTPD tick: {error}"))?;
            let _recovery = connection.recovery_gate.lock().await;
            connection
                .portfolio
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

pub async fn recover_and_synchronize(
    client: &Client,
    ctpd: &CtpdConfig,
    bearer_token: &str,
    instruments: &[InstrumentConfig],
    portfolio: &Arc<RwLock<Portfolio>>,
) -> Result<(), String> {
    let now_ms = chrono::Utc::now().timestamp_millis();
    let mut recovered = Vec::new();
    for instrument in instruments {
        recovered.extend(
            recover_completed_klines(client, ctpd, bearer_token, instrument, portfolio, now_ms)
                .await?,
        );
    }
    let recovered_count = {
        let mut portfolio = portfolio.write().expect("portfolio lock poisoned");
        let recovered_count = apply_recovered_klines(&mut portfolio, &mut recovered)?;
        for instrument in instruments {
            portfolio.mark_market_synchronized(&instrument.market_instrument_id)?;
        }
        recovered_count
    };
    if recovered_count > 0 {
        eprintln!("bullet-live: recovered {recovered_count} completed CTPD Klines");
    }
    Ok(())
}

#[derive(Debug)]
struct RecoveredKline {
    market_instrument_id: String,
    bar: HistoryBar,
}

async fn recover_completed_klines(
    client: &Client,
    ctpd: &CtpdConfig,
    bearer_token: &str,
    instrument: &InstrumentConfig,
    portfolio: &Arc<RwLock<Portfolio>>,
    now_ms: i64,
) -> Result<Vec<RecoveredKline>, String> {
    let last_market_ns = portfolio
        .read()
        .expect("portfolio lock poisoned")
        .last_bar_timestamp_ns(&instrument.market_instrument_id)
        .ok_or("cannot recover CTPD Klines before Parquet history is seeded")?;
    let last_ms = ctpd_ms_from_timestamp_ns(last_market_ns)?;
    if now_ms < last_ms {
        return Err("CTPD clock precedes the seeded Parquet history".into());
    }
    let mut recovered = Vec::new();
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
            recovered.push(RecoveredKline {
                market_instrument_id: instrument.market_instrument_id.clone(),
                bar,
            });
        }
    }
    Ok(recovered)
}

fn apply_recovered_klines(
    portfolio: &mut Portfolio,
    recovered: &mut [RecoveredKline],
) -> Result<usize, String> {
    sort_recovered_klines(recovered);
    let mut applied = 0_usize;
    let mut batch_start = 0_usize;
    while batch_start < recovered.len() {
        let timestamp_ns = recovered[batch_start].bar.timestamp_ns;
        let mut batch_end = batch_start;
        let mut batch_applied = false;
        while batch_end < recovered.len() && recovered[batch_end].bar.timestamp_ns == timestamp_ns {
            let recovered_kline = &recovered[batch_end];
            if portfolio.ingest_recovered_history(
                &recovered_kline.market_instrument_id,
                &recovered_kline.bar,
            )? {
                applied += 1;
                batch_applied = true;
            }
            batch_end += 1;
        }
        if batch_applied {
            portfolio.flush_history_candidates();
        }
        batch_start = batch_end;
    }
    Ok(applied)
}

fn sort_recovered_klines(recovered: &mut [RecoveredKline]) {
    recovered.sort_by(|left, right| {
        left.bar
            .timestamp_ns
            .cmp(&right.bar.timestamp_ns)
            .then_with(|| left.market_instrument_id.cmp(&right.market_instrument_id))
    });
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
    let timestamp_ns = timestamp_ns_from_ctpd_ms(
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
        collections::HashMap,
        future::IntoFuture,
        path::PathBuf,
        sync::{Arc, RwLock},
    };

    use axum::{
        Json, Router,
        extract::{Query, State},
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::get,
    };
    use chrono::{Duration, NaiveDateTime, TimeZone, Utc};
    use reqwest::Client;
    use tokio::sync::Mutex;

    use super::{
        CtpdConnection, CtpdKline, RecoveredKline, SseDecoder, consume_connection,
        history_bar_from_kline, kline_recovery_windows, recover_and_synchronize,
        sort_recovered_klines,
    };
    use crate::{
        config::{CtpdConfig, InstrumentConfig},
        market_time::{ctpd_ms_from_timestamp_ns, timestamp_ns_from_shanghai_wall_clock},
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
    fn keeps_ctpd_utc_klines_as_real_instants() {
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
        let expected = timestamp_ns_from_shanghai_wall_clock(
            NaiveDateTime::parse_from_str("20260810 09:31:00", "%Y%m%d %H:%M:%S").unwrap(),
        )
        .unwrap();
        assert_eq!(bar.timestamp_ns, expected);
        assert_eq!(
            ctpd_ms_from_timestamp_ns(expected).unwrap(),
            start_ms + 60_000
        );
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

    #[test]
    fn recognizes_the_cffex_lunch_break() {
        let at = |hour, minute| Utc.with_ymd_and_hms(2026, 8, 11, hour, minute, 0).unwrap();

        assert!(!super::is_cffex_lunch_break(at(3, 29)));
        assert!(super::is_cffex_lunch_break(at(3, 30)));
        assert!(super::is_cffex_lunch_break(at(4, 59)));
        assert!(!super::is_cffex_lunch_break(at(5, 0)));
    }

    #[test]
    fn orders_same_timestamp_recovery_by_market_identifier() {
        let bar = |timestamp_ns| HistoryBar {
            timestamp_ns,
            open: 4_000.0,
            high: 4_000.0,
            low: 4_000.0,
            close: 4_000.0,
            volume: 1.0,
            money: 4_000.0,
            open_interest: 1.0,
        };
        let mut recovered = vec![
            RecoveredKline {
                market_instrument_id: "IDX-CFFEX-IM".into(),
                bar: bar(2),
            },
            RecoveredKline {
                market_instrument_id: "IDX-CFFEX-IH".into(),
                bar: bar(1),
            },
            RecoveredKline {
                market_instrument_id: "IDX-CFFEX-IF".into(),
                bar: bar(1),
            },
            RecoveredKline {
                market_instrument_id: "IDX-CFFEX-IC".into(),
                bar: bar(1),
            },
        ];

        sort_recovered_klines(&mut recovered);

        assert_eq!(
            recovered
                .iter()
                .map(|kline| (kline.bar.timestamp_ns, kline.market_instrument_id.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (1, "IDX-CFFEX-IC"),
                (1, "IDX-CFFEX-IF"),
                (1, "IDX-CFFEX-IH"),
                (2, "IDX-CFFEX-IM"),
            ]
        );
    }

    #[tokio::test]
    async fn synchronizes_all_markets_after_one_timestamp_ordered_recovery() {
        let server = Router::new().route("/v1/klines", get(mock_recovery_klines));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_task = tokio::spawn(axum::serve(listener, server).into_future());
        let instruments = vec![
            instrument("IC8888", "IDX-CFFEX-IC", "IC2609"),
            instrument("IF8888", "IDX-CFFEX-IF", "IF2609"),
            instrument("IH8888", "IDX-CFFEX-IH", "IH2609"),
            instrument("IM8888", "IDX-CFFEX-IM", "IM2609"),
        ];
        let timestamp = |time| {
            timestamp_ns_from_shanghai_wall_clock(
                NaiveDateTime::parse_from_str(time, "%Y%m%d %H:%M:%S").unwrap(),
            )
            .unwrap()
        };
        let mut initial = Portfolio::default();
        for instrument in &instruments {
            initial.insert(
                instrument.market_instrument_id.clone(),
                Portfolio::new_model(instrument),
            );
            initial
                .ingest_history(
                    &instrument.market_instrument_id,
                    &HistoryBar {
                        timestamp_ns: timestamp("20260810 09:30:00"),
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
        }
        let portfolio = Arc::new(RwLock::new(initial));

        recover_and_synchronize(
            &Client::new(),
            &CtpdConfig {
                base_url: format!("http://{address}"),
                bearer_token_file: PathBuf::from("/unused"),
                stale_after_ms: 1_000,
            },
            "ctpd-test-token",
            &instruments,
            &portfolio,
        )
        .await
        .unwrap();
        server_task.abort();

        let portfolio = portfolio.read().unwrap();
        assert!(portfolio.is_fully_synchronized());
        for instrument in &instruments {
            assert_eq!(
                portfolio.last_bar_timestamp_ns(&instrument.market_instrument_id),
                Some(timestamp("20260810 09:31:00"))
            );
        }
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
                    timestamp_ns: timestamp_ns_from_shanghai_wall_clock(
                        NaiveDateTime::parse_from_str("20260810 09:30:00", "%Y%m%d %H:%M:%S")
                            .unwrap(),
                    )
                    .unwrap(),
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
        let client = Client::new();
        let ctpd = CtpdConfig {
            base_url: format!("http://{address}"),
            bearer_token_file: PathBuf::from("/unused"),
            stale_after_ms: 1_000,
        };
        let recovery_gate = Arc::new(Mutex::new(()));
        let connection = CtpdConnection {
            client: &client,
            ctpd: &ctpd,
            bearer_token: "ctpd-test-token",
            instruments: std::slice::from_ref(&instrument),
            portfolio: &portfolio,
            recovery_gate: &recovery_gate,
        };
        let error = consume_connection(&connection, &instrument, false)
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

    async fn mock_recovery_klines(
        headers: HeaderMap,
        Query(query): Query<HashMap<String, String>>,
    ) -> impl IntoResponse {
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
        let price = match query.get("instrument_id").map(String::as_str) {
            Some("IDX-CFFEX-IC") => 6_000.0,
            Some("IDX-CFFEX-IF") => 4_000.0,
            Some("IDX-CFFEX-IH") => 3_000.0,
            Some("IDX-CFFEX-IM") => 7_000.0,
            _ => return (StatusCode::BAD_REQUEST, Json(Vec::new())),
        };
        (
            StatusCode::OK,
            Json(vec![serde_json::json!({
                "start_ms": 1786325400000_i64,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1,
                "money": price,
                "open_interest": 1.0,
                "closed": true,
            })]),
        )
    }

    fn instrument(symbol: &str, market: &str, target: &str) -> InstrumentConfig {
        InstrumentConfig {
            symbol: symbol.into(),
            market_instrument_id: market.into(),
            target_instrument_id: target.into(),
            parquet: "/unused".into(),
            full_weight_contracts: 10.0,
            contract_multiplier: 300.0,
            session_bar_count: 240,
            session_end_time: "15:00:00".into(),
        }
    }
}

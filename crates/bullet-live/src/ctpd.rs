use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use futures_util::StreamExt;
use reqwest::Client;
use tokio::time::{sleep, timeout};

use crate::{
    config::{CtpdConfig, InstrumentConfig},
    model::{CtpdTick, Portfolio},
};

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
                instrument.ctpd_instrument_id
            );
        }
        portfolio
            .write()
            .expect("portfolio lock poisoned")
            .clear(&instrument.ctpd_instrument_id);
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
    let url = format!(
        "{}/v1/ticks?instrument_id={}",
        ctpd.base_url.trim_end_matches('/'),
        instrument.ctpd_instrument_id
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
        Router,
        extract::State,
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::get,
    };
    use chrono::{Duration, NaiveDateTime};
    use reqwest::Client;

    use super::{SseDecoder, consume_connection};
    use crate::{
        config::{CtpdConfig, InstrumentConfig},
        model::{Lab0344Model, Portfolio},
    };

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

    #[tokio::test]
    async fn consumes_ctpd_sse_and_publishes_a_live_target() {
        let payload = (0..61)
            .map(|minute| {
                let at = NaiveDateTime::parse_from_str("20260810 09:30:00", "%Y%m%d %H:%M:%S")
                    .unwrap()
                    + Duration::minutes(minute);
                format!(
                    "event: tick\ndata: {{\"instrument_id\":\"IF2609\",\"exchange_id\":\"CFFEX\",\"trading_day\":\"{}\",\"action_day\":\"{}\",\"update_time\":\"{}\",\"update_millisec\":0,\"last_price\":{},\"open_interest\":{}}}\n\n",
                    at.format("%Y%m%d"),
                    at.format("%Y%m%d"),
                    at.format("%H:%M:%S"),
                    4_000 + minute,
                    minute + 1,
                )
            })
            .collect::<String>();
        let server = Router::new()
            .route("/v1/ticks", get(mock_ticks))
            .with_state(payload);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server_task = tokio::spawn(axum::serve(listener, server).into_future());
        let instrument = InstrumentConfig {
            symbol: "IF".into(),
            ctpd_instrument_id: "IF2609".into(),
            parquet: "/unused".into(),
            target_contracts: 1,
            contract_multiplier: 300.0,
            session_bar_count: 240,
            last_executable_signal_time: "14:40:00".into(),
        };
        let mut initial = Portfolio::default();
        initial.insert("IF2609".into(), Lab0344Model::new(&instrument));
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
        assert_eq!(portfolio.read().unwrap().targets()[0].contracts, 1);
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
}

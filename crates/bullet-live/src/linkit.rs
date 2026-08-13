use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use reqwest::Client;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::{config::LinkitConfig, market_time::shanghai, model::LiveTradeSignal};

const DELIVERY_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Serialize)]
struct DirectMessage<'a> {
    recipient_username: &'a str,
    body: String,
}

pub async fn send_loop(
    client: Client,
    config: LinkitConfig,
    bearer_token: String,
    mut receiver: mpsc::Receiver<LiveTradeSignal>,
) {
    while let Some(signal) = receiver.recv().await {
        if let Err(error) = send_signal(&client, &config, &bearer_token, &signal).await {
            eprintln!(
                "bullet-live: Linkit notification {} {} failed: {error}",
                signal.action, signal.candidate_id
            );
        }
    }
}

async fn send_signal(
    client: &Client,
    config: &LinkitConfig,
    bearer_token: &str,
    signal: &LiveTradeSignal,
) -> Result<(), String> {
    let url = format!("{}/bot/v1/messages", config.base_url.trim_end_matches('/'));
    client
        .post(url)
        .bearer_auth(bearer_token)
        .json(&DirectMessage {
            recipient_username: &config.recipient_username,
            body: format_signal(signal)?,
        })
        .timeout(DELIVERY_TIMEOUT)
        .send()
        .await
        .map_err(|error| format!("request failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Linkit rejected message: {error}"))?;
    Ok(())
}

fn format_signal(signal: &LiveTradeSignal) -> Result<String, String> {
    let at_ns = i64::try_from(signal.at_ns).map_err(|_| "signal timestamp outside chrono range")?;
    let at = DateTime::<Utc>::from_timestamp_nanos(at_ns)
        .with_timezone(&shanghai())
        .to_rfc3339_opts(SecondsFormat::Secs, false);
    Ok(format!(
        "Bullet lab0334 {action}: {side} {contracts} {instrument} @ {price:.2} ({symbol}, {at})",
        action = signal.action,
        side = signal.side,
        contracts = signal.contracts,
        instrument = signal.target_instrument_id,
        price = signal.price,
        symbol = signal.symbol,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::{
        Json, Router,
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::post,
    };
    use serde_json::Value;

    use super::{LinkitConfig, format_signal, send_signal};
    use crate::model::LiveTradeSignal;

    type CapturedRequest = Arc<Mutex<Option<(String, Value)>>>;

    fn signal() -> LiveTradeSignal {
        LiveTradeSignal {
            candidate_id: "IM8888-1786654800000000000-1".into(),
            symbol: "IM8888".into(),
            target_instrument_id: "IM2609".into(),
            side: "LONG".into(),
            action: "OPEN",
            contracts: 3.0,
            price: 7_123.5,
            at_ns: 1_786_654_800_000_000_000,
        }
    }

    #[test]
    fn renders_a_shanghai_timestamp() {
        assert_eq!(
            format_signal(&signal()).unwrap(),
            "Bullet lab0334 OPEN: LONG 3 IM2609 @ 7123.50 (IM8888, 2026-08-14T05:00:00+08:00)"
        );
    }

    #[tokio::test]
    async fn posts_the_documented_direct_message_payload() {
        let captured = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route("/bot/v1/messages", post(capture))
            .with_state(captured.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(axum::serve(listener, app).into_future());
        let config = LinkitConfig {
            base_url: format!("http://{address}"),
            bearer_token_file: "/unused".into(),
            recipient_username: "0xCZ".into(),
        };

        send_signal(
            &reqwest::Client::new(),
            &config,
            "linkit-test-token",
            &signal(),
        )
        .await
        .unwrap();
        server.abort();

        let (authorization, payload) = captured.lock().unwrap().clone().unwrap();
        assert_eq!(authorization, "Bearer linkit-test-token");
        assert_eq!(payload["recipient_username"], "0xCZ");
        assert_eq!(
            payload["body"],
            "Bullet lab0334 OPEN: LONG 3 IM2609 @ 7123.50 (IM8888, 2026-08-14T05:00:00+08:00)"
        );
    }

    #[tokio::test]
    async fn reports_a_rejected_message_without_touching_the_signal() {
        let app = Router::new().route(
            "/bot/v1/messages",
            post(|| async { StatusCode::SERVICE_UNAVAILABLE }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(axum::serve(listener, app).into_future());
        let config = LinkitConfig {
            base_url: format!("http://{address}"),
            bearer_token_file: "/unused".into(),
            recipient_username: "0xCZ".into(),
        };
        let signal = signal();

        assert!(
            send_signal(&reqwest::Client::new(), &config, "test", &signal)
                .await
                .is_err()
        );
        assert_eq!(signal, super::tests::signal());
        server.abort();
    }

    async fn capture(
        State(captured): State<CapturedRequest>,
        headers: HeaderMap,
        Json(payload): Json<Value>,
    ) -> StatusCode {
        *captured.lock().unwrap() = Some((
            headers
                .get("authorization")
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned(),
            payload,
        ));
        StatusCode::OK
    }
}

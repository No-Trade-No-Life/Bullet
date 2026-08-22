use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use reqwest::Client;
use serde::Serialize;
use tokio::{sync::mpsc, time::sleep};

use crate::{config::LinkitConfig, market_time::shanghai, model::LiveTradeSignal};

const DELIVERY_TIMEOUT: Duration = Duration::from_secs(3);
const LINKIT_ORIGIN: &str = "https://linkit.ntnl.io";

#[derive(Serialize)]
struct GroupMessage<'a> {
    conversation_id: &'a str,
    client_message_id: &'a str,
    body: String,
}

#[derive(serde::Deserialize)]
struct BotConversation {
    kind: String,
}

pub async fn validate_group(
    client: &Client,
    config: &LinkitConfig,
    bearer_token: &str,
) -> Result<(), String> {
    let response = client
        .get(format!(
            "{LINKIT_ORIGIN}/bot/v1/conversations/{}",
            config.conversation_id
        ))
        .bearer_auth(bearer_token)
        .timeout(DELIVERY_TIMEOUT)
        .send()
        .await
        .map_err(|error| format!("group validation request failed: {error}"))?
        .error_for_status()
        .map_err(|error| format!("Linkit rejected target group: {error}"))?
        .json::<BotConversation>()
        .await
        .map_err(|error| format!("invalid Linkit group response: {error}"))?;
    (response.kind == "group")
        .then_some(())
        .ok_or_else(|| "Linkit target conversation is not a group".to_owned())
}

pub async fn send_loop(
    client: Client,
    config: LinkitConfig,
    bearer_token: String,
    mut receiver: mpsc::Receiver<LiveTradeSignal>,
) {
    while let Some(signal) = receiver.recv().await {
        loop {
            match send_signal(&client, &config, &bearer_token, &signal).await {
                Ok(()) => break,
                Err(error) => {
                    eprintln!(
                        "bullet-live: Linkit delivery pending for signal {}: {error}",
                        signal.candidate_id
                    );
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }
    }
}

async fn send_signal(
    client: &Client,
    config: &LinkitConfig,
    bearer_token: &str,
    signal: &LiveTradeSignal,
) -> Result<(), String> {
    send_signal_to(client, LINKIT_ORIGIN, config, bearer_token, signal).await
}

async fn send_signal_to(
    client: &Client,
    origin: &str,
    config: &LinkitConfig,
    bearer_token: &str,
    signal: &LiveTradeSignal,
) -> Result<(), String> {
    let url = format!("{}/bot/v1/messages", origin.trim_end_matches('/'));
    client
        .post(url)
        .bearer_auth(bearer_token)
        .json(&GroupMessage {
            conversation_id: &config.conversation_id,
            client_message_id: &signal.candidate_id,
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
        "Bullet live simulated target · lab0334\nSignal: {action} {side} {contracts} {instrument} @ {price:.2}\nSource: current CTPD live feed · execution: simulated target only, no broker order\nStrategy: lab0334 · symbol: {symbol} · time: {at}\nSignal ID: {candidate_id}",
        action = signal.action,
        side = signal.side,
        contracts = signal.contracts,
        instrument = signal.target_instrument_id,
        price = signal.price,
        symbol = signal.symbol,
        candidate_id = signal.candidate_id,
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

    use super::{LinkitConfig, format_signal, send_signal_to};
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
            "Bullet live simulated target · lab0334\nSignal: OPEN LONG 3 IM2609 @ 7123.50\nSource: current CTPD live feed · execution: simulated target only, no broker order\nStrategy: lab0334 · symbol: IM8888 · time: 2026-08-14T05:00:00+08:00\nSignal ID: IM8888-1786654800000000000-1"
        );
    }

    #[tokio::test]
    async fn posts_the_documented_group_message_payload() {
        let captured = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route("/bot/v1/messages", post(capture))
            .with_state(captured.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(axum::serve(listener, app).into_future());
        let config = LinkitConfig {
            bearer_token_file: "/unused".into(),
            conversation_id: "5077b76d-962f-45dd-83dd-05e78b5cabd7".into(),
        };

        send_signal_to(
            &reqwest::Client::new(),
            &format!("http://{address}"),
            &config,
            "linkit-test-token",
            &signal(),
        )
        .await
        .unwrap();
        server.abort();

        let (authorization, payload) = captured.lock().unwrap().clone().unwrap();
        assert_eq!(authorization, "Bearer linkit-test-token");
        assert_eq!(
            payload["conversation_id"],
            "5077b76d-962f-45dd-83dd-05e78b5cabd7"
        );
        assert!(payload.get("recipient_username").is_none());
        assert_eq!(payload["client_message_id"], "IM8888-1786654800000000000-1");
        assert_eq!(
            payload["body"],
            "Bullet live simulated target · lab0334\nSignal: OPEN LONG 3 IM2609 @ 7123.50\nSource: current CTPD live feed · execution: simulated target only, no broker order\nStrategy: lab0334 · symbol: IM8888 · time: 2026-08-14T05:00:00+08:00\nSignal ID: IM8888-1786654800000000000-1"
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
            bearer_token_file: "/unused".into(),
            conversation_id: "5077b76d-962f-45dd-83dd-05e78b5cabd7".into(),
        };
        let signal = signal();

        assert!(
            send_signal_to(
                &reqwest::Client::new(),
                &format!("http://{address}"),
                &config,
                "test",
                &signal
            )
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

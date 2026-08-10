use std::sync::{Arc, RwLock};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::get,
};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;

use crate::model::{Portfolio, TargetPosition};

#[derive(Clone)]
pub struct RemoteAccountState {
    pub account_id: String,
    pub bearer_token: Option<String>,
    pub portfolio: Arc<RwLock<Portfolio>>,
}

pub fn app(state: RemoteAccountState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/api/accounts", get(accounts))
        .route("/api/positions", get(positions))
        .with_state(state)
}

async fn healthz() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn accounts(
    State(state): State<RemoteAccountState>,
    headers: HeaderMap,
) -> Result<Json<Vec<AccountMeta>>, StatusCode> {
    authorize(&headers, &state.bearer_token)?;
    Ok(Json(vec![AccountMeta {
        account_id: state.account_id,
    }]))
}

#[derive(Deserialize)]
struct PositionQuery {
    account_id: String,
}

async fn positions(
    State(state): State<RemoteAccountState>,
    headers: HeaderMap,
    Query(query): Query<PositionQuery>,
) -> Result<Json<Vec<Position>>, StatusCode> {
    authorize(&headers, &state.bearer_token)?;
    if query.account_id != state.account_id {
        return Ok(Json(vec![]));
    }
    let targets = state
        .portfolio
        .read()
        .expect("portfolio lock poisoned")
        .targets();
    Ok(Json(
        targets
            .iter()
            .map(|target| Position::from_target(&state.account_id, target))
            .collect(),
    ))
}

fn authorize(headers: &HeaderMap, expected: &Option<String>) -> Result<(), StatusCode> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let Some(actual) = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
    else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    let expected = format!("Bearer {expected}");
    (expected.len() == actual.len() && expected.as_bytes().ct_eq(actual.as_bytes()).into())
        .then_some(())
        .ok_or(StatusCode::UNAUTHORIZED)
}

#[derive(Serialize)]
struct AccountMeta {
    account_id: String,
}

#[derive(Serialize)]
struct Position {
    position_id: String,
    product_id: String,
    account_id: String,
    updated_at: i64,
    base_currency: Option<String>,
    quote_currency: Option<String>,
    amount: f64,
    position_price: f64,
    closable_price: f64,
    current_price: String,
    notional_value: f64,
    notional_currency: Option<String>,
    settlement_currency: String,
    notional: String,
    valuation: f64,
    floating_profit: f64,
    comment: String,
    margin: Option<f64>,
}

impl Position {
    fn from_target(account_id: &str, target: &TargetPosition) -> Self {
        let amount = target.contracts as f64;
        let notional_value = amount.abs() * target.latest_price * target.multiplier;
        let floating_profit =
            (target.latest_price - target.entry_price) * amount * target.multiplier;
        Self {
            position_id: format!("BULLET/FUTURES/{}", target.symbol),
            product_id: format!("BULLET/FUTURES/{}", target.ctpd_instrument_id),
            account_id: account_id.to_owned(),
            updated_at: target.updated_at_ms,
            base_currency: Some(target.symbol.clone()),
            quote_currency: Some("CNY".into()),
            amount,
            position_price: target.entry_price,
            closable_price: target.latest_price,
            current_price: target.latest_price.to_string(),
            notional_value,
            notional_currency: Some("CNY".into()),
            settlement_currency: "CNY".into(),
            notional: (if amount.is_sign_negative() {
                -notional_value
            } else {
                notional_value
            })
            .to_string(),
            valuation: floating_profit,
            floating_profit,
            comment: "lab-0344 process_exception_not_freezable; simulated target only".into(),
            margin: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use axum::{body::Body, http::Request};
    use chrono::{Duration, NaiveDateTime};
    use tower::ServiceExt;

    use super::{RemoteAccountState, app};
    use crate::{
        config::InstrumentConfig,
        model::{CtpdTick, Lab0344Model, Portfolio},
    };

    fn seeded_portfolio() -> Arc<RwLock<Portfolio>> {
        let config = InstrumentConfig {
            symbol: "IF".into(),
            ctpd_instrument_id: "IF2609".into(),
            parquet: "/unused".into(),
            target_contracts: 1,
            contract_multiplier: 300.0,
            session_bar_count: 240,
            last_executable_signal_time: "14:40:00".into(),
        };
        let model = Lab0344Model::new(&config);
        let mut portfolio = Portfolio::default();
        portfolio.insert("IF2609".into(), model);
        for minute in 0..61 {
            let at = NaiveDateTime::parse_from_str("20260810 09:30:00", "%Y%m%d %H:%M:%S").unwrap()
                + Duration::minutes(minute);
            portfolio
                .ingest(CtpdTick {
                    instrument_id: "IF2609".into(),
                    exchange_id: "CFFEX".into(),
                    trading_day: at.format("%Y%m%d").to_string(),
                    action_day: at.format("%Y%m%d").to_string(),
                    update_time: at.format("%H:%M:%S").to_string(),
                    update_millisec: 0,
                    last_price: 4_000.0 + minute as f64,
                    open_interest: minute as f64 + 1.0,
                })
                .unwrap();
        }
        Arc::new(RwLock::new(portfolio))
    }

    #[tokio::test]
    async fn remote_protocol_authenticates_and_returns_unknown_account_as_empty() {
        let service = app(RemoteAccountState {
            account_id: "BULLET/lab0344".into(),
            bearer_token: Some("secret".into()),
            portfolio: seeded_portfolio(),
        });
        let unauthorized = service
            .clone()
            .oneshot(Request::get("/api/accounts").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), 401);
        let accounts = service
            .clone()
            .oneshot(
                Request::get("/api/accounts")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accounts.status(), 200);
        let body = axum::body::to_bytes(accounts.into_body(), 1024)
            .await
            .unwrap();
        let accounts: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            accounts,
            serde_json::json!([{"account_id":"BULLET/lab0344"}])
        );
        let positions = service
            .clone()
            .oneshot(
                Request::get("/api/positions?account_id=BULLET%2Flab0344")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(positions.status(), 200);
        let body = axum::body::to_bytes(positions.into_body(), 4096)
            .await
            .unwrap();
        let positions: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(positions[0]["account_id"], "BULLET/lab0344");
        assert_eq!(positions[0]["amount"], 1.0);
        assert_eq!(positions[0]["settlement_currency"], "CNY");
        assert!(
            positions[0]["floating_profit"]
                .as_f64()
                .unwrap()
                .is_finite()
        );
        let unknown = service
            .oneshot(
                Request::get("/api/positions?account_id=other")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unknown.status(), 200);
        let body = axum::body::to_bytes(unknown.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(body, "[]");
    }
}

use std::sync::{Arc, RwLock};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::get,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;

use crate::model::{HistoricalFill, HistoryWindow, Portfolio, TargetPosition};

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
        .route("/api/account-history", get(account_history))
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

#[derive(Deserialize)]
struct AccountHistoryQuery {
    account_id: String,
    cursor: Option<String>,
    limit: Option<usize>,
    start_at: Option<String>,
    end_at: Option<String>,
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
            .filter_map(|target| Position::from_target(&state.account_id, target))
            .collect(),
    ))
}

async fn account_history(
    State(state): State<RemoteAccountState>,
    headers: HeaderMap,
    Query(query): Query<AccountHistoryQuery>,
) -> Result<Json<AccountHistory>, StatusCode> {
    authorize(&headers, &state.bearer_token)?;
    if query.account_id != state.account_id {
        return Err(StatusCode::BAD_REQUEST);
    }
    let portfolio = state.portfolio.read().expect("portfolio lock poisoned");
    let history = page_history(
        &state.account_id,
        portfolio.history_fills(),
        portfolio.history_window(),
        &query,
    )?;
    Ok(Json(history))
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
struct AccountHistory {
    account_id: String,
    record_type: &'static str,
    records: Vec<TradeFill>,
    coverage: Vec<HistoryCoverage>,
    next_cursor: Option<String>,
}

#[derive(Serialize)]
struct TradeFill {
    account_id: String,
    exchange: &'static str,
    trade_id: String,
    order_id: Option<String>,
    product_id: String,
    direction: &'static str,
    price: f64,
    amount: f64,
    value: f64,
    value_currency: &'static str,
    fee: f64,
    fee_currency: &'static str,
    created_at: String,
}

#[derive(Debug, Serialize, PartialEq)]
struct HistoryCoverage {
    account_id: String,
    start_at: Option<String>,
    end_at: Option<String>,
    complete: bool,
    detail: &'static str,
}

#[derive(Deserialize, Serialize)]
struct HistoryCursor {
    coverage_start_at_ns: Option<u64>,
    coverage_end_at_ns: Option<u64>,
    filter_start_at_ns: Option<u64>,
    filter_end_at_ns: Option<u64>,
    created_at_ns: u64,
    trade_id: String,
}

const HISTORY_DETAIL: &str = "Bullet reconstructs lab0334 simulated fills from the configured Parquet seed and completed CTPD bars; this is not a broker execution ledger";

fn page_history(
    account_id: &str,
    mut fills: Vec<HistoricalFill>,
    window: HistoryWindow,
    query: &AccountHistoryQuery,
) -> Result<AccountHistory, StatusCode> {
    let cursor = query.cursor.as_deref().map(decode_cursor).transpose()?;
    let filter_start_at_ns = query.start_at.as_deref().map(parse_timestamp).transpose()?;
    let filter_end_at_ns = query.end_at.as_deref().map(parse_timestamp).transpose()?;
    if filter_start_at_ns
        .zip(filter_end_at_ns)
        .is_some_and(|(start, end)| start > end)
    {
        return Err(StatusCode::BAD_REQUEST);
    }
    let snapshot = cursor.unwrap_or(HistoryCursor {
        coverage_start_at_ns: window.start_at_ns,
        coverage_end_at_ns: window.end_at_ns,
        filter_start_at_ns,
        filter_end_at_ns,
        created_at_ns: u64::MAX,
        trade_id: "\u{10ffff}".into(),
    });
    fills.retain(|fill| {
        snapshot
            .coverage_end_at_ns
            .is_none_or(|end| fill.created_at_ns <= end)
            && snapshot
                .filter_start_at_ns
                .is_none_or(|start| fill.created_at_ns >= start)
            && snapshot
                .filter_end_at_ns
                .is_none_or(|end| fill.created_at_ns <= end)
    });
    fills.sort_by(|left, right| {
        right
            .created_at_ns
            .cmp(&left.created_at_ns)
            .then_with(|| right.trade_id.cmp(&left.trade_id))
    });
    if query.cursor.is_some() {
        fills.retain(|fill| {
            (fill.created_at_ns, fill.trade_id.as_str())
                < (snapshot.created_at_ns, snapshot.trade_id.as_str())
        });
    }
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let next_cursor = (fills.len() > limit)
        .then(|| {
            let last = &fills[limit - 1];
            encode_cursor(HistoryCursor {
                coverage_start_at_ns: snapshot.coverage_start_at_ns,
                coverage_end_at_ns: snapshot.coverage_end_at_ns,
                filter_start_at_ns: snapshot.filter_start_at_ns,
                filter_end_at_ns: snapshot.filter_end_at_ns,
                created_at_ns: last.created_at_ns,
                trade_id: last.trade_id.clone(),
            })
        })
        .transpose()?;
    fills.truncate(limit);
    let records = fills
        .into_iter()
        .map(|fill| {
            Ok(TradeFill {
                account_id: account_id.into(),
                exchange: "CFFEX",
                trade_id: fill.trade_id,
                order_id: Some(fill.order_id),
                product_id: fill.product_id,
                direction: fill.direction.as_protocol_value(),
                price: fill.price,
                amount: fill.amount,
                value: fill.value,
                value_currency: "CNY",
                fee: fill.fee,
                fee_currency: "CNY",
                created_at: format_timestamp(fill.created_at_ns)?,
            })
        })
        .collect::<Result<Vec<_>, StatusCode>>()?;
    Ok(AccountHistory {
        account_id: account_id.into(),
        record_type: "TRADE_FILL_V1",
        records,
        coverage: vec![HistoryCoverage {
            account_id: account_id.into(),
            start_at: snapshot
                .coverage_start_at_ns
                .map(format_timestamp)
                .transpose()?,
            end_at: snapshot
                .coverage_end_at_ns
                .map(format_timestamp)
                .transpose()?,
            complete: false,
            detail: HISTORY_DETAIL,
        }],
        next_cursor,
    })
}

fn parse_timestamp(value: &str) -> Result<u64, StatusCode> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .and_then(|time| time.timestamp_nanos_opt())
        .and_then(|time| u64::try_from(time).ok())
        .ok_or(StatusCode::BAD_REQUEST)
}

fn format_timestamp(timestamp_ns: u64) -> Result<String, StatusCode> {
    i64::try_from(timestamp_ns)
        .ok()
        .map(DateTime::<Utc>::from_timestamp_nanos)
        .map(|time| time.to_rfc3339_opts(SecondsFormat::Nanos, true))
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)
}

fn encode_cursor(cursor: HistoryCursor) -> Result<String, StatusCode> {
    serde_json::to_vec(&cursor)
        .map(|value| URL_SAFE_NO_PAD.encode(value))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn decode_cursor(cursor: &str) -> Result<HistoryCursor, StatusCode> {
    URL_SAFE_NO_PAD
        .decode(cursor)
        .ok()
        .and_then(|value| serde_json::from_slice(&value).ok())
        .ok_or(StatusCode::BAD_REQUEST)
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
    fn from_target(account_id: &str, target: &TargetPosition) -> Option<Self> {
        let amount = target.contracts;
        let notional_value = amount * target.latest_price * target.multiplier;
        let floating_profit =
            (target.latest_price - target.entry_price) * amount * target.multiplier;
        if !amount.is_finite()
            || !target.entry_price.is_finite()
            || !target.latest_price.is_finite()
            || !target.multiplier.is_finite()
            || !notional_value.is_finite()
            || !floating_profit.is_finite()
        {
            return None;
        }
        Some(Self {
            position_id: format!("BULLET/FUTURES/{}", target.symbol),
            product_id: format!("BULLET/FUTURES/{}", target.target_instrument_id),
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
            notional: notional_value.to_string(),
            // Futures exposure is not cash equity. 1Exchange values derivative
            // positions by the NAV-additive unrealized PnL instead of the
            // gross notional, which would otherwise overstate this target-only
            // simulated account by several orders of magnitude.
            valuation: floating_profit,
            floating_profit,
            comment: "lab-0334 trusted baseline; simulated target only, no order-routing".into(),
            margin: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    use super::{AccountHistoryQuery, Position, RemoteAccountState, app, page_history};
    use crate::model::{FillDirection, HistoricalFill, HistoryWindow, Portfolio, TargetPosition};

    fn seeded_portfolio() -> Arc<RwLock<Portfolio>> {
        Arc::new(RwLock::new(Portfolio::default()))
    }

    #[tokio::test]
    async fn remote_protocol_authenticates_and_returns_unknown_account_as_empty() {
        let service = app(RemoteAccountState {
            account_id: "BULLET/lab0334".into(),
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
            serde_json::json!([{"account_id":"BULLET/lab0334"}])
        );
        let positions = service
            .clone()
            .oneshot(
                Request::get("/api/positions?account_id=BULLET%2Flab0334")
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
        assert_eq!(positions, serde_json::json!([]));
        let history_unauthorized = service
            .clone()
            .oneshot(
                Request::get("/api/account-history?account_id=BULLET%2Flab0334")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(history_unauthorized.status(), 401);
        let history = service
            .clone()
            .oneshot(
                Request::get("/api/account-history?account_id=BULLET%2Flab0334")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(history.status(), 200);
        let body = axum::body::to_bytes(history.into_body(), 4096)
            .await
            .unwrap();
        let history: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(history["record_type"], "TRADE_FILL_V1");
        assert_eq!(history["records"], serde_json::json!([]));
        assert_eq!(history["coverage"].as_array().unwrap().len(), 1);
        assert!(history["next_cursor"].is_null());
        let unknown_history = service
            .clone()
            .oneshot(
                Request::get("/api/account-history?account_id=other")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unknown_history.status(), 400);
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

    #[test]
    fn account_history_cursor_keeps_coverage_stable_without_duplicates() {
        let fills: Vec<_> = (1..=3)
            .map(|at| HistoricalFill {
                trade_id: format!("fill-{at}"),
                order_id: format!("order-{at}"),
                product_id: "BULLET/FUTURES/IM2609".into(),
                direction: FillDirection::Long,
                price: 4_000.0,
                amount: 1.0,
                value: 1_200_000.0,
                fee: 0.0,
                created_at_ns: at,
            })
            .collect();
        let first = page_history(
            "BULLET/lab0334",
            fills.clone(),
            HistoryWindow {
                start_at_ns: Some(1),
                end_at_ns: Some(3),
            },
            &AccountHistoryQuery {
                account_id: "BULLET/lab0334".into(),
                cursor: None,
                limit: Some(2),
                start_at: None,
                end_at: None,
            },
        )
        .unwrap();
        let second = page_history(
            "BULLET/lab0334",
            fills,
            HistoryWindow {
                start_at_ns: Some(0),
                end_at_ns: Some(4),
            },
            &AccountHistoryQuery {
                account_id: "BULLET/lab0334".into(),
                cursor: first.next_cursor.clone(),
                limit: Some(2),
                start_at: None,
                end_at: None,
            },
        )
        .unwrap();

        assert_eq!(
            first
                .records
                .iter()
                .map(|record| record.trade_id.as_str())
                .collect::<Vec<_>>(),
            vec!["fill-3", "fill-2"]
        );
        assert_eq!(
            second
                .records
                .iter()
                .map(|record| record.trade_id.as_str())
                .collect::<Vec<_>>(),
            vec!["fill-1"]
        );
        assert_eq!(first.coverage, second.coverage);
        assert!(second.next_cursor.is_none());
    }

    #[test]
    fn serializes_short_exposure_and_positive_profit_consistently() {
        let position = Position::from_target(
            "BULLET/lab0334",
            &TargetPosition {
                symbol: "IF8888".into(),
                target_instrument_id: "IF2609".into(),
                exchange_id: "CFFEX".into(),
                contracts: -3.0,
                entry_price: 4_000.0,
                latest_price: 3_990.0,
                multiplier: 300.0,
                updated_at_ms: 1,
            },
        )
        .unwrap();
        assert!(position.amount.is_sign_negative());
        assert!(position.notional_value.is_sign_negative());
        assert!(position.valuation.is_sign_positive());
        assert!(position.floating_profit.is_sign_positive());
    }
}

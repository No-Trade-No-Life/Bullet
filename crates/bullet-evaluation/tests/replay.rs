use bullet_evaluation::{
    Accounting, CausalAvailability, EvaluationConfig, EvaluationInput, EventTime, MarketPoint,
    TargetDecision, TerminalPolicy, evaluate,
};

fn time(timestamp_ns: u64) -> EventTime {
    EventTime {
        timestamp_ns,
        sequence: 0,
    }
}

fn config() -> EvaluationConfig {
    EvaluationConfig {
        accounting: Accounting::NormalizedExposureV1,
        instrument: "BTC".to_owned(),
        one_way_cost_bps: 10.0,
        slippage_bps: 0.0,
        terminal_policy: TerminalPolicy::Liquidate,
        sharpe_periods_per_year: 1.0,
        annualization_days_per_year: 1.0,
        evaluation_days: vec![0],
    }
}

fn decision(timestamp_ns: u64, target_units: i64, provenance: &str) -> TargetDecision {
    TargetDecision {
        decision_time: time(timestamp_ns - 1),
        execution_time: time(timestamp_ns),
        instrument: "BTC".to_owned(),
        target_units,
        model_vintage: Some("fixture-v1".to_owned()),
        signal_id: Some("signal".to_owned()),
        source_id: Some("fixture".to_owned()),
        state_code: Some("ready".to_owned()),
        diagnostic_json: serde_json::json!({"b": 2, "a": 1}),
        causal_availability: CausalAvailability {
            source_end: time(timestamp_ns - 2),
            available_at: time(timestamp_ns - 1),
        },
        provenance_hash: provenance.to_owned(),
    }
}

fn input() -> EvaluationInput {
    EvaluationInput {
        schema_version: 1,
        config: config(),
        market: vec![
            MarketPoint {
                time: time(100),
                instrument: "BTC".to_owned(),
                price: 100.0,
            },
            MarketPoint {
                time: time(101),
                instrument: "BTC".to_owned(),
                price: 100.0,
            },
            MarketPoint {
                time: time(102),
                instrument: "BTC".to_owned(),
                price: 110.0,
            },
            MarketPoint {
                time: time(103),
                instrument: "BTC".to_owned(),
                price: 100.0,
            },
        ],
        decisions: vec![
            decision(
                101,
                1,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            decision(
                102,
                -1,
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
        ],
    }
}

#[test]
fn replays_reversal_and_terminal_liquidation_as_separate_executions() {
    let result = evaluate(&input()).expect("valid normalized-exposure input");

    assert_eq!(result.execution_ledger.len(), 3);
    assert_eq!(result.execution_ledger[0].delta_units, 1);
    assert_eq!(result.execution_ledger[1].delta_units, -2);
    assert_eq!(result.execution_ledger[2].delta_units, 1);
    assert_eq!(result.metrics.turnover_units, 4);
    assert_eq!(result.metrics.ending_target_units, -1);
    assert_eq!(result.metrics.ending_realized_units, 0);
    assert_eq!(result.position_ledger.len(), 3);
    assert_eq!(result.daily_returns.len(), 1);
    assert!(result.metrics.total_return_units > 0);
}

#[test]
fn rejects_first_market_point_on_an_unregistered_day() {
    let mut value = input();
    value.market[0].time = time(86_400_000_000_000);
    let error = evaluate(&value).expect_err("the first point must also be on the calendar");

    assert_eq!(error.to_string(), "unregistered realization day");
}

#[test]
fn rejects_same_time_decisions_without_a_sequence_order() {
    let mut value = input();
    let mut second = decision(
        2,
        0,
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    );
    second.execution_time = time(101);
    second.decision_time = time(100);
    second.causal_availability = CausalAvailability {
        source_end: time(98),
        available_at: time(99),
    };
    value.decisions[1] = second;
    let error = evaluate(&value).expect_err("execution keys must be unique");

    assert_eq!(
        error.to_string(),
        "execution keys must be unique and strictly ordered"
    );
}

#[test]
fn fixed_format_matches_the_existing_twelve_decimal_contract() {
    assert_eq!(bullet_evaluation::to_fixed(0.5e-12).unwrap(), 0);
    assert_eq!(bullet_evaluation::to_fixed(1.5e-12).unwrap(), 2);
    assert_eq!(bullet_evaluation::to_fixed(2.5e-12).unwrap(), 2);
    assert_eq!(bullet_evaluation::to_fixed(-0.5e-12).unwrap(), 0);
    assert_eq!(bullet_evaluation::to_fixed(-1.5e-12).unwrap(), -2);
}

use crate::*;

#[derive(Clone)]
pub(crate) struct DayAccumulator {
    pub(crate) gross_factor: f64,
    pub(crate) net_factor: f64,
    pub(crate) cost: i64,
    pub(crate) turnover: u64,
}

pub(crate) fn add(left: i64, right: i64) -> Result<i64> {
    left.checked_add(right)
        .ok_or_else(|| EvaluationError("canonical sum overflows".into()))
}
pub(crate) fn add_turnover(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| EvaluationError("turnover overflows".into()))
}

pub(super) fn replay(input: &EvaluationInput) -> Result<EvaluationResult> {
    let config = &input.config;
    let mut days = vec![
        DayAccumulator {
            gross_factor: 1.0,
            net_factor: 1.0,
            cost: 0,
            turnover: 0
        };
        config.evaluation_days.len()
    ];
    let mut decision_ledger = Vec::with_capacity(input.decisions.len());
    let mut execution_ledger = Vec::new();
    let mut position_ledger = Vec::with_capacity(input.market.len() - 1);
    let mut units = 0_i64;
    let mut decision_index = 0;
    let mut completed_trades = 0;
    let mut ending_target = 0;
    let mut total_turnover = 0;
    let mut total_cost = 0;

    for (index, points) in input.market.windows(2).enumerate() {
        let start = &points[0];
        let end = &points[1];
        let mut turnover = 0;
        if let Some(decision) = input.decisions.get(decision_index)
            && decision.execution_time == start.time
        {
            let delta = decision
                .target_units
                .checked_sub(units)
                .ok_or_else(|| EvaluationError("target delta overflows".into()))?;
            turnover = delta.unsigned_abs();
            if delta != 0 {
                execution_ledger.push(execution(
                    start,
                    units,
                    decision.target_units,
                    Some(decision_index),
                    ExecutionReason::Rebalance,
                    config,
                )?);
            }
            if units != 0 && units.signum() != decision.target_units.signum() {
                completed_trades += 1;
            }
            units = decision.target_units;
            let mut canonical_decision = decision.clone();
            canonical_decision.diagnostic_json.sort_all_objects();
            decision_ledger.push(DecisionLedgerRow {
                decision: canonical_decision,
                realized_units: units,
            });
            decision_index += 1;
        }
        ending_target = units;
        let liquidation = index + 2 == input.market.len()
            && config.terminal_policy == TerminalPolicy::Liquidate
            && units != 0;
        if liquidation {
            turnover = add_turnover(turnover, units.unsigned_abs())?;
            execution_ledger.push(execution(
                end,
                units,
                0,
                None,
                ExecutionReason::TerminalLiquidation,
                config,
            )?);
            completed_trades += 1;
        }
        let gross = units as f64 * (end.price / start.price - 1.0);
        let cost = turnover as f64 * (config.one_way_cost_bps + config.slippage_bps) / 10_000.0;
        let net = gross - cost;
        require(
            gross.is_finite() && net.is_finite() && net > -1.0,
            format!("interval {index} has non-finite return or exhausted equity"),
        )?;
        let cost_units = to_fixed(cost)?;
        let day = end.time.timestamp_ns / NANOS_PER_DAY * NANOS_PER_DAY;
        let day_index = ((day - config.evaluation_days[0]) / NANOS_PER_DAY) as usize;
        let accumulator = &mut days[day_index];
        accumulator.gross_factor *= 1.0 + gross;
        accumulator.net_factor *= 1.0 + net;
        accumulator.cost = add(accumulator.cost, cost_units)?;
        accumulator.turnover = add_turnover(accumulator.turnover, turnover)?;
        total_turnover = add_turnover(total_turnover, turnover)?;
        total_cost = add(total_cost, cost_units)?;
        position_ledger.push(PositionLedgerRow {
            start_time: start.time,
            end_time: end.time,
            instrument: config.instrument.clone(),
            realized_units: units,
            turnover_units: turnover,
            gross_return_units: to_fixed(gross)?,
            cost_units,
            net_return_units: to_fixed(net)?,
        });
        if liquidation {
            units = 0;
        }
    }
    let daily_returns = build_daily_returns(&days, config)?;
    let mut metrics = build_metrics(&days, config)?;
    metrics.turnover_units = total_turnover;
    metrics.total_cost_units = total_cost;
    metrics.completed_directional_trades = completed_trades;
    metrics.ending_target_units = ending_target;
    metrics.ending_realized_units = units;
    let config_sha256 = audit::hash(config)?;
    let market_sha256 = audit::hash(&input.market)?;
    // Hash canonical (recursively sorted) diagnostics, not caller insertion order.
    let decision_sha256 = audit::hash(
        &decision_ledger
            .iter()
            .map(|row| &row.decision)
            .collect::<Vec<_>>(),
    )?;
    let ledger_sha256 = audit::hash(&(
        &decision_ledger,
        &execution_ledger,
        &position_ledger,
        &daily_returns,
        &metrics,
    ))?;
    let run_sha256 = audit::hash(&(
        SCHEMA_VERSION,
        &config_sha256,
        &market_sha256,
        &decision_sha256,
        &ledger_sha256,
    ))?;
    let audit = AuditManifest {
        schema_version: SCHEMA_VERSION,
        market_point_count: input.market.len(),
        decision_count: decision_ledger.len(),
        execution_count: execution_ledger.len(),
        interval_count: position_ledger.len(),
        config_sha256,
        market_sha256,
        decision_sha256,
        ledger_sha256,
        run_sha256,
    };
    Ok(EvaluationResult {
        schema_version: SCHEMA_VERSION,
        config: config.clone(),
        decision_ledger,
        execution_ledger,
        position_ledger,
        daily_returns,
        metrics,
        audit,
    })
}

pub(crate) fn execution(
    point: &MarketPoint,
    from_units: i64,
    realized_units: i64,
    decision_index: Option<usize>,
    reason: ExecutionReason,
    config: &EvaluationConfig,
) -> Result<ExecutionLedgerRow> {
    let delta_units = realized_units
        .checked_sub(from_units)
        .ok_or_else(|| EvaluationError("target delta overflows".into()))?;
    Ok(ExecutionLedgerRow {
        execution_time: point.time,
        instrument: point.instrument.clone(),
        decision_index,
        reason,
        from_units,
        realized_units,
        delta_units,
        price: point.price,
        cost_units: to_fixed(
            delta_units.unsigned_abs() as f64 * (config.one_way_cost_bps + config.slippage_bps)
                / 10_000.0,
        )?,
    })
}

pub(crate) fn build_daily_returns(
    days: &[DayAccumulator],
    config: &EvaluationConfig,
) -> Result<Vec<DailyReturnRow>> {
    config
        .evaluation_days
        .iter()
        .zip(days)
        .map(|(day, value)| {
            Ok(DailyReturnRow {
                day_timestamp_ns: *day,
                gross_return_units: to_fixed(value.gross_factor - 1.0)?,
                net_return_units: to_fixed(value.net_factor - 1.0)?,
                cost_sum_units: value.cost,
                turnover_units: value.turnover,
            })
        })
        .collect()
}

pub(crate) fn build_metrics(
    days: &[DayAccumulator],
    config: &EvaluationConfig,
) -> Result<EvaluationMetrics> {
    let returns: Vec<_> = days.iter().map(|value| value.net_factor - 1.0).collect();
    let count = returns.len();
    let mean = returns.iter().sum::<f64>() / count as f64;
    let volatility = if count > 1 {
        (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (count - 1) as f64).sqrt()
    } else {
        0.0
    };
    let sharpe = if volatility > 0.0 {
        mean / volatility * config.sharpe_periods_per_year.sqrt()
    } else {
        0.0
    };
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut drawdown = 0.0_f64;
    for value in &returns {
        equity *= 1.0 + value;
        peak = peak.max(equity);
        drawdown = drawdown.max(1.0 - equity / peak);
    }
    Ok(EvaluationMetrics {
        days: count,
        total_return_units: to_fixed(equity - 1.0)?,
        mean_daily_return_units: to_fixed(mean)?,
        daily_volatility_units: to_fixed(volatility)?,
        sharpe_units: to_fixed(sharpe)?,
        annualized_return_units: to_fixed(
            equity.powf(config.annualization_days_per_year / count as f64) - 1.0,
        )?,
        max_drawdown_units: to_fixed(drawdown)?,
        turnover_units: 0,
        total_cost_units: 0,
        completed_directional_trades: 0,
        ending_target_units: 0,
        ending_realized_units: 0,
    })
}

use crate::*;

pub(crate) fn validate_config(config: &EvaluationConfig) -> Result<()> {
    require(
        !config.instrument.trim().is_empty(),
        "instrument must not be empty",
    )?;
    for value in [config.one_way_cost_bps, config.slippage_bps] {
        require(
            value.is_finite() && value >= 0.0,
            "costs must be finite and nonnegative",
        )?;
    }
    require(
        (config.one_way_cost_bps + config.slippage_bps).is_finite(),
        "cost sum overflows",
    )?;
    require(
        matches!(config.sharpe_standard_deviation_ddof, 0 | 1),
        "Sharpe standard deviation ddof must be 0 or 1",
    )?;
    for value in [
        config.sharpe_periods_per_year,
        config.annualization_days_per_year,
    ] {
        require(
            value.is_finite() && value > 0.0,
            "annualization must be finite and positive",
        )?;
    }
    require(
        !config.evaluation_days.is_empty(),
        "evaluation calendar is empty",
    )?;
    for (index, day) in config.evaluation_days.iter().enumerate() {
        require(
            day % NANOS_PER_DAY == 0,
            "calendar days must be UTC midnights",
        )?;
        if index > 0 {
            require(
                config.evaluation_days[index - 1].checked_add(NANOS_PER_DAY) == Some(*day),
                "calendar days must be consecutive and strictly ordered",
            )?;
        }
    }
    Ok(())
}

pub(super) fn validate(input: &EvaluationInput) -> Result<()> {
    require(
        input.schema_version == SCHEMA_VERSION,
        "unsupported schema version",
    )?;
    validate_config(&input.config)?;
    let config = &input.config;
    require(
        input.market.len() >= 2,
        "at least two market points are required",
    )?;
    for (index, point) in input.market.iter().enumerate() {
        require(
            point.instrument == config.instrument,
            "market instrument differs from config",
        )?;
        require(
            point.price.is_finite() && point.price > 0.0,
            "market prices must be finite and positive",
        )?;
        let day = point.time.timestamp_ns / NANOS_PER_DAY * NANOS_PER_DAY;
        require(
            config.evaluation_days.binary_search(&day).is_ok(),
            "unregistered realization day",
        )?;
        if index > 0 {
            require(
                input.market[index - 1].time.timestamp_ns < point.time.timestamp_ns,
                "market timestamps must be strictly increasing",
            )?;
        }
    }
    let mut cursor = 0;
    let mut previous_decision_time = None;
    for (index, decision) in input.decisions.iter().enumerate() {
        require(
            decision.instrument == config.instrument,
            "decision instrument differs from config",
        )?;
        // Integer-to-binary64 conversion must not silently change the exposure.
        require(
            decision.target_units.unsigned_abs() <= (1_u64 << 53),
            "target exceeds exact binary64 integer range",
        )?;
        require(
            decision.provenance_hash.len() == 64
                && decision
                    .provenance_hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "provenance_hash must be lowercase SHA-256 hex",
        )?;
        let availability = &decision.causal_availability;
        require(
            availability.source_end <= availability.available_at
                && availability.available_at <= decision.decision_time
                && decision.decision_time < decision.execution_time,
            "source/availability/decision/fill ordering is non-causal",
        )?;
        require(
            previous_decision_time.is_none_or(|time| time <= decision.decision_time),
            "decision times are not ordered",
        )?;
        previous_decision_time = Some(decision.decision_time);
        if index > 0 {
            require(
                input.decisions[index - 1].execution_time < decision.execution_time,
                "execution keys must be unique and strictly ordered",
            )?;
        }
        while cursor < input.market.len() - 1 && input.market[cursor].time < decision.execution_time
        {
            cursor += 1;
        }
        require(
            cursor < input.market.len() - 1 && input.market[cursor].time == decision.execution_time,
            "decision has no matching nonterminal execution point",
        )?;
    }
    Ok(())
}

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    io,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::model::{CandidateDecision, CandidateLabel, Portfolio};

const DECIMAL_PLACES: usize = 14;

/// The replay verifier compares the entire causal candidate ledger, not only
/// the final accepted targets. Python and Rust floating-point reductions may
/// differ beneath the declared 14-decimal precision, so values are normalized
/// before the canonical JSONL byte comparison. Branching is compared before
/// normalization through every decision field.
#[derive(Clone, Copy, Debug)]
pub struct ParitySummary {
    pub decisions: usize,
    pub labels: usize,
    pub canonical_bytes: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct ReferenceDecision {
    candidate_id: String,
    symbol: String,
    side: String,
    trade_type: String,
    candidate_policy_id: String,
    planned_exit_policy: String,
    prediction_asof_time: String,
    prediction_key: String,
    candidate_pred: String,
    history_count: String,
    history_max_label_available_time: String,
    decision: String,
    reject_reason: String,
    active_count: String,
    used_weight: String,
    candidate_weight: String,
    same_symbol_count: String,
    incumbent_trade_id: String,
    incumbent_candidate_policy_id: String,
    incumbent_pred: String,
    replacement_margin: String,
    capital_ok: String,
    symbol_ok: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ReferenceLabel {
    symbol: String,
    side: String,
    trade_type: String,
    candidate_policy_id: String,
    entry_time: String,
    entry_price: String,
    label_available_time: String,
    exit_time: String,
    exit_price: String,
    trade_return: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct CandidateIdentity {
    symbol: String,
    time: String,
    side: String,
    trade_type: String,
    candidate_policy_id: String,
}

impl CandidateIdentity {
    fn reference(decision: &ReferenceDecision) -> Self {
        Self {
            symbol: decision.symbol.clone(),
            time: decision.prediction_asof_time.clone(),
            side: decision.side.clone(),
            trade_type: decision.trade_type.clone(),
            candidate_policy_id: decision.candidate_policy_id.clone(),
        }
    }

    fn bullet(decision: &CandidateDecision) -> Result<Self, String> {
        Ok(Self {
            symbol: decision.symbol.clone(),
            time: format_timestamp(decision.prediction_asof_ns)?,
            side: decision.side.clone(),
            trade_type: decision.trade_type.clone(),
            candidate_policy_id: decision.candidate_policy_id.clone(),
        })
    }

    fn label_reference(label: &ReferenceLabel) -> Self {
        Self {
            symbol: label.symbol.clone(),
            time: label.entry_time.clone(),
            side: label.side.clone(),
            trade_type: label.trade_type.clone(),
            candidate_policy_id: label.candidate_policy_id.clone(),
        }
    }

    fn label_bullet(label: &CandidateLabel) -> Result<Self, String> {
        Ok(Self {
            symbol: label.symbol.clone(),
            time: format_timestamp(label.entry_time_ns)?,
            side: label.side.clone(),
            trade_type: label.trade_type.clone(),
            candidate_policy_id: label.candidate_policy_id.clone(),
        })
    }
}

#[derive(Debug, Serialize)]
struct CanonicalDecision {
    candidate: CandidateIdentity,
    planned_exit_policy: String,
    prediction_key: String,
    candidate_pred: String,
    history_count: String,
    history_max_label_available_time: String,
    decision: String,
    reject_reason: String,
    active_count: String,
    used_weight: String,
    candidate_weight: String,
    same_symbol_count: String,
    incumbent_candidate: String,
    incumbent_candidate_policy_id: String,
    incumbent_pred: String,
    replacement_margin: String,
    capital_ok: String,
    symbol_ok: String,
}

#[derive(Debug, Serialize)]
struct CanonicalLabel {
    candidate: CandidateIdentity,
    entry_price: String,
    label_available_time: String,
    exit_time: String,
    exit_price: String,
    trade_return: String,
}

pub fn verify(
    portfolio: &Portfolio,
    reference_decisions_path: &str,
    reference_labels_path: &str,
) -> Result<ParitySummary, Box<dyn Error>> {
    verify_inner(portfolio, reference_decisions_path, reference_labels_path)
        .map_err(|error| Box::new(io::Error::other(error)) as Box<dyn Error>)
}

fn verify_inner(
    portfolio: &Portfolio,
    reference_decisions_path: &str,
    reference_labels_path: &str,
) -> Result<ParitySummary, String> {
    let reference_decisions = read_csv(reference_decisions_path)?;
    let reference_labels = read_csv(reference_labels_path)?;
    let reference_decision_bytes = canonical_reference_decisions(&reference_decisions)?;
    let bullet_decision_bytes = canonical_bullet_decisions(portfolio.candidate_decisions())?;
    compare(
        "decisions",
        &reference_decision_bytes,
        &bullet_decision_bytes,
    )?;

    let reference_label_bytes = canonical_reference_labels(&reference_labels)?;
    let bullet_label_bytes = canonical_bullet_labels(portfolio.candidate_labels())?;
    compare("labels", &reference_label_bytes, &bullet_label_bytes)?;

    Ok(ParitySummary {
        decisions: portfolio.candidate_decisions().len(),
        labels: portfolio.candidate_labels().len(),
        canonical_bytes: reference_decision_bytes.len() + reference_label_bytes.len(),
    })
}

fn read_csv<T: DeserializeOwned>(path: &str) -> Result<Vec<T>, String> {
    let mut reader =
        csv::Reader::from_path(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    reader
        .deserialize()
        .collect::<Result<Vec<T>, _>>()
        .map_err(|error| format!("cannot parse {path}: {error}"))
}

fn canonical_reference_decisions(records: &[ReferenceDecision]) -> Result<Vec<u8>, String> {
    let identities = reference_decision_identities(records)?;
    let rows = records
        .iter()
        .map(|record| {
            let incumbent_candidate = optional_identity(&record.incumbent_trade_id, &identities)?;
            Ok(CanonicalDecision {
                candidate: CandidateIdentity::reference(record),
                planned_exit_policy: record.planned_exit_policy.clone(),
                prediction_key: record.prediction_key.clone(),
                candidate_pred: decimal(&record.candidate_pred)?,
                history_count: unsigned(&record.history_count)?,
                history_max_label_available_time: record.history_max_label_available_time.clone(),
                decision: record.decision.clone(),
                reject_reason: record.reject_reason.clone(),
                active_count: unsigned(&record.active_count)?,
                used_weight: decimal(&record.used_weight)?,
                candidate_weight: decimal(&record.candidate_weight)?,
                same_symbol_count: unsigned(&record.same_symbol_count)?,
                incumbent_candidate,
                incumbent_candidate_policy_id: record.incumbent_candidate_policy_id.clone(),
                incumbent_pred: optional_decimal(&record.incumbent_pred)?,
                replacement_margin: optional_decimal(&record.replacement_margin)?,
                capital_ok: boolean(&record.capital_ok)?,
                symbol_ok: boolean(&record.symbol_ok)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    canonical_jsonl(rows)
}

fn canonical_bullet_decisions(records: &[CandidateDecision]) -> Result<Vec<u8>, String> {
    let identities = bullet_decision_identities(records)?;
    let rows = records
        .iter()
        .map(|record| {
            let incumbent_candidate = optional_identity(
                record.incumbent_candidate_id.as_deref().unwrap_or_default(),
                &identities,
            )?;
            Ok(CanonicalDecision {
                candidate: CandidateIdentity::bullet(record)?,
                planned_exit_policy: record.planned_exit_policy.clone(),
                prediction_key: record.prediction_key.clone(),
                candidate_pred: decimal_number(record.candidate_pred)?,
                history_count: record.history_count.to_string(),
                history_max_label_available_time: optional_timestamp(
                    record.history_max_label_available_ns,
                )?,
                decision: record.decision.clone(),
                reject_reason: record.reject_reason.clone().unwrap_or_default(),
                active_count: record.active_count.to_string(),
                used_weight: decimal_number(record.used_weight)?,
                candidate_weight: decimal_number(record.candidate_weight)?,
                same_symbol_count: record.same_symbol_count.to_string(),
                incumbent_candidate,
                incumbent_candidate_policy_id: record
                    .incumbent_candidate_policy_id
                    .clone()
                    .unwrap_or_default(),
                incumbent_pred: optional_number(record.incumbent_pred)?,
                replacement_margin: optional_number(record.replacement_margin)?,
                capital_ok: optional_bool(record.capital_ok),
                symbol_ok: optional_bool(record.symbol_ok),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    canonical_jsonl(rows)
}

fn canonical_reference_labels(records: &[ReferenceLabel]) -> Result<Vec<u8>, String> {
    let rows = records
        .iter()
        .map(|record| {
            Ok(CanonicalLabel {
                candidate: CandidateIdentity::label_reference(record),
                entry_price: decimal(&record.entry_price)?,
                label_available_time: record.label_available_time.clone(),
                exit_time: record.exit_time.clone(),
                exit_price: decimal(&record.exit_price)?,
                trade_return: decimal(&record.trade_return)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    canonical_jsonl(rows)
}

fn canonical_bullet_labels(records: &[CandidateLabel]) -> Result<Vec<u8>, String> {
    let rows = records
        .iter()
        .map(|record| {
            let exit_time = format_timestamp(record.label_available_ns)?;
            Ok(CanonicalLabel {
                candidate: CandidateIdentity::label_bullet(record)?,
                entry_price: decimal_number(record.entry_price)?,
                label_available_time: exit_time.clone(),
                exit_time,
                exit_price: decimal_number(record.exit_price)?,
                trade_return: decimal_number(record.trade_return)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    canonical_jsonl(rows)
}

fn reference_decision_identities(
    records: &[ReferenceDecision],
) -> Result<BTreeMap<String, CandidateIdentity>, String> {
    let mut identities = BTreeMap::new();
    for record in records {
        if identities
            .insert(
                record.candidate_id.clone(),
                CandidateIdentity::reference(record),
            )
            .is_some()
        {
            return Err("reference decisions contain duplicate candidate_id values".into());
        }
    }
    Ok(identities)
}

fn bullet_decision_identities(
    records: &[CandidateDecision],
) -> Result<BTreeMap<String, CandidateIdentity>, String> {
    let mut identities = BTreeMap::new();
    for record in records {
        let identity = CandidateIdentity::bullet(record)?;
        if identities
            .insert(record.candidate_id.clone(), identity)
            .is_some()
        {
            return Err("Bullet decisions contain duplicate candidate_id values".into());
        }
    }
    Ok(identities)
}

fn optional_identity(
    candidate_id: &str,
    identities: &BTreeMap<String, CandidateIdentity>,
) -> Result<String, String> {
    if candidate_id.is_empty() {
        return Ok(String::new());
    }
    let identity = identities.get(candidate_id).ok_or_else(|| {
        format!("incumbent candidate {candidate_id} is absent from the decision ledger")
    })?;
    serde_json::to_string(identity).map_err(|error| error.to_string())
}

fn canonical_jsonl<T: Serialize>(records: Vec<T>) -> Result<Vec<u8>, String> {
    let mut lines = records
        .iter()
        .map(|record| serde_json::to_string(record).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let unique = lines.iter().collect::<BTreeSet<_>>().len();
    if unique != lines.len() {
        return Err("canonical ledger contains duplicate records".into());
    }
    lines.sort_unstable();
    Ok(format!("{}\n", lines.join("\n")).into_bytes())
}

fn compare(name: &str, reference: &[u8], bullet: &[u8]) -> Result<(), String> {
    if reference == bullet {
        return Ok(());
    }
    let reference_lines = String::from_utf8_lossy(reference);
    let bullet_lines = String::from_utf8_lossy(bullet);
    let line = reference_lines
        .lines()
        .zip(bullet_lines.lines())
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| {
            reference_lines
                .lines()
                .count()
                .min(bullet_lines.lines().count())
        });
    let reference_line = reference_lines.lines().nth(line).unwrap_or("<end>");
    let bullet_line = bullet_lines.lines().nth(line).unwrap_or("<end>");
    Err(format!(
        "lab0334 parity failed for {name} at canonical line {}: reference={reference_line} bullet={bullet_line}",
        line + 1
    ))
}

fn decimal(value: &str) -> Result<String, String> {
    value
        .parse::<f64>()
        .map_err(|error| format!("invalid decimal {value:?}: {error}"))
        .and_then(decimal_number)
}

fn optional_decimal(value: &str) -> Result<String, String> {
    if value.is_empty() {
        Ok(String::new())
    } else {
        decimal(value)
    }
}

fn decimal_number(value: f64) -> Result<String, String> {
    if !value.is_finite() {
        return Err("canonical ledger contains a non-finite decimal".into());
    }
    let normalized = if value == 0.0 { 0.0 } else { value };
    Ok(format!("{normalized:.DECIMAL_PLACES$}"))
}

fn optional_number(value: Option<f64>) -> Result<String, String> {
    value
        .map(decimal_number)
        .transpose()
        .map(Option::unwrap_or_default)
}

fn unsigned(value: &str) -> Result<String, String> {
    value
        .parse::<usize>()
        .map(|number| number.to_string())
        .map_err(|error| format!("invalid unsigned integer {value:?}: {error}"))
}

fn boolean(value: &str) -> Result<String, String> {
    if value.is_empty() {
        return Ok(String::new());
    }
    match value {
        "True" | "true" => Ok("true".into()),
        "False" | "false" => Ok("false".into()),
        _ => Err(format!("invalid boolean {value:?}")),
    }
}

fn optional_bool(value: Option<bool>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn optional_timestamp(value: Option<u64>) -> Result<String, String> {
    value
        .map(format_timestamp)
        .transpose()
        .map(Option::unwrap_or_default)
}

fn format_timestamp(value: u64) -> Result<String, String> {
    let timestamp = i64::try_from(value).map_err(|_| "timestamp is outside chrono range")?;
    Ok(DateTime::<Utc>::from_timestamp_nanos(timestamp)
        .naive_utc()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::decimal_number;

    #[test]
    fn canonical_decimal_removes_negative_zero() {
        assert_eq!(decimal_number(-0.0).unwrap(), "0.00000000000000");
        assert_eq!(decimal_number(0.125).unwrap(), "0.12500000000000");
    }
}

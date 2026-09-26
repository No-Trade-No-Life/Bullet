use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::accounting::{
    DayAccumulator, add, add_turnover, build_daily_returns, build_metrics, execution,
};
use crate::audit;
use crate::{
    CausalAvailability, DailyReturnRow, DecisionLedgerRow, EvaluationConfig, EvaluationError,
    EvaluationMetrics, EventTime, ExecutionReason, MarketPoint, NANOS_PER_DAY, PositionLedgerRow,
    Result, SCHEMA_VERSION, TargetDecision, TerminalPolicy, to_fixed,
};

const DECISION_FILE: &str = "decision_ledger.jsonl";
const EXECUTION_FILE: &str = "execution_ledger.jsonl";
const POSITION_FILE: &str = "position_ledger.jsonl";
const DAILY_FILE: &str = "daily_returns.jsonl";
const CHECKPOINT_FILE: &str = "checkpoint.json";
const SUMMARY_FILE: &str = "summary.json";

#[derive(Clone, Debug)]
pub struct JsonlReplayOptions {
    pub config: EvaluationConfig,
    pub market_path: PathBuf,
    pub decisions_path: PathBuf,
    pub output_dir: PathBuf,
    pub market_sha256: String,
    pub decision_sha256: String,
    pub checkpoint_every_intervals: usize,
    pub stop_after_intervals: Option<usize>,
    pub resume: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamStatus {
    Complete,
    CheckpointOnly,
}

#[derive(Clone, Debug, Serialize)]
pub struct StreamOutputManifest {
    pub decision_ledger: String,
    pub execution_ledger: String,
    pub position_ledger: String,
    pub daily_returns: String,
    pub checkpoint: String,
    pub summary: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct StreamRunSummary {
    pub schema_version: u32,
    pub status: StreamStatus,
    pub processed_intervals: usize,
    pub market_point_count: usize,
    pub decision_count: usize,
    pub execution_count: usize,
    pub output: StreamOutputManifest,
    pub metrics: Option<EvaluationMetrics>,
    pub audit: Option<crate::AuditManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputFileState {
    bytes: u64,
    lines: usize,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OutputCheckpoint {
    decision: OutputFileState,
    execution: OutputFileState,
    position: OutputFileState,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MarketCheckpoint {
    time: EventTime,
    instrument: String,
    price_bits: u64,
}

impl MarketCheckpoint {
    fn from_point(point: &MarketPoint) -> Self {
        Self {
            time: point.time,
            instrument: point.instrument.clone(),
            price_bits: point.price.to_bits(),
        }
    }

    fn to_point(&self) -> MarketPoint {
        MarketPoint {
            time: self.time,
            instrument: self.instrument.clone(),
            price: f64::from_bits(self.price_bits),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DayCheckpoint {
    gross_factor_bits: u64,
    net_factor_bits: u64,
    cost: i64,
    turnover: u64,
}

impl From<&DayAccumulator> for DayCheckpoint {
    fn from(value: &DayAccumulator) -> Self {
        Self {
            gross_factor_bits: value.gross_factor.to_bits(),
            net_factor_bits: value.net_factor.to_bits(),
            cost: value.cost,
            turnover: value.turnover,
        }
    }
}

impl DayCheckpoint {
    fn into_day(self) -> DayAccumulator {
        DayAccumulator {
            gross_factor: f64::from_bits(self.gross_factor_bits),
            net_factor: f64::from_bits(self.net_factor_bits),
            cost: self.cost,
            turnover: self.turnover,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StreamCheckpoint {
    schema_version: u32,
    config_sha256: String,
    market_sha256: String,
    decision_sha256: String,
    processed_intervals: usize,
    market_point_count: usize,
    consumed_decisions: usize,
    units: i64,
    ending_target_units: i64,
    completed_directional_trades: usize,
    total_turnover_units: u64,
    total_cost_units: i64,
    last_decision_time: Option<EventTime>,
    last_execution_time: Option<EventTime>,
    last_market: MarketCheckpoint,
    days: Vec<DayCheckpoint>,
    outputs: OutputCheckpoint,
    completed: bool,
}

struct JsonlReader<T> {
    lines: std::io::Lines<BufReader<File>>,
    buffered: Option<T>,
    index: usize,
    path: PathBuf,
    marker: PhantomData<T>,
}

impl<T: DeserializeOwned> JsonlReader<T> {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|error| io_error(path, error))?;
        Ok(Self {
            lines: BufReader::new(file).lines(),
            buffered: None,
            index: 0,
            path: path.to_owned(),
            marker: PhantomData,
        })
    }

    fn read_next(&mut self) -> Result<Option<T>> {
        let Some(line) = self.lines.next() else {
            return Ok(None);
        };
        let line = line.map_err(|error| io_error(&self.path, error))?;
        self.index += 1;
        serde_json::from_str(&line).map(Some).map_err(|error| {
            EvaluationError(format!(
                "cannot parse {} line {}: {error}",
                self.path.display(),
                self.index
            ))
        })
    }

    fn next(&mut self) -> Result<Option<T>> {
        self.buffered
            .take()
            .map_or_else(|| self.read_next(), |value| Ok(Some(value)))
    }

    fn is_empty(&mut self) -> Result<bool> {
        if self.buffered.is_none() {
            self.buffered = self.read_next()?;
        }
        Ok(self.buffered.is_none())
    }

    fn skip(&mut self, count: usize) -> Result<()> {
        for _ in 0..count {
            if self.next()?.is_none() {
                return Err(EvaluationError(format!(
                    "{} ended before skip count {count}",
                    self.path.display()
                )));
            }
        }
        Ok(())
    }
}

struct TrackedFile {
    writer: BufWriter<File>,
    path: PathBuf,
    bytes: u64,
    lines: usize,
    hasher: Sha256,
}

impl TrackedFile {
    fn create(path: PathBuf) -> Result<Self> {
        let file = File::create_new(&path).map_err(|error| io_error(&path, error))?;
        Ok(Self {
            writer: BufWriter::new(file),
            path,
            bytes: 0,
            lines: 0,
            hasher: Sha256::new(),
        })
    }

    fn resume(path: PathBuf, state: &OutputFileState) -> Result<Self> {
        let actual = file_state(&path)?;
        if actual.bytes != state.bytes
            || actual.lines != state.lines
            || actual.sha256 != state.sha256
        {
            return Err(EvaluationError(format!(
                "output prefix differs from checkpoint: {}",
                path.display()
            )));
        }
        let mut bytes = Vec::new();
        File::open(&path)
            .map_err(|error| io_error(&path, error))?
            .read_to_end(&mut bytes)
            .map_err(|error| io_error(&path, error))?;
        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|error| io_error(&path, error))?;
        file.flush().map_err(|error| io_error(&path, error))?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        Ok(Self {
            writer: BufWriter::new(file),
            path,
            bytes: state.bytes,
            lines: state.lines,
            hasher,
        })
    }

    fn write<T: Serialize>(&mut self, value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value)
            .map_err(|error| EvaluationError(format!("cannot serialize ledger row: {error}")))?;
        self.writer
            .write_all(&bytes)
            .and_then(|_| self.writer.write_all(b"\n"))
            .map_err(|error| io_error(&self.path, error))?;
        self.hasher.update(&bytes);
        self.hasher.update(b"\n");
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64 + 1)
            .ok_or_else(|| EvaluationError("ledger byte count overflows".into()))?;
        self.lines += 1;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|error| io_error(&self.path, error))
    }

    fn state(&self) -> OutputFileState {
        OutputFileState {
            bytes: self.bytes,
            lines: self.lines,
            sha256: hex(&self.hasher.clone().finalize()),
        }
    }
}

struct LedgerFiles {
    decisions: TrackedFile,
    executions: TrackedFile,
    positions: TrackedFile,
}

impl LedgerFiles {
    fn new(directory: &Path, checkpoint: Option<&OutputCheckpoint>) -> Result<Self> {
        fs::create_dir_all(directory).map_err(|error| io_error(directory, error))?;
        let paths = (
            directory.join(DECISION_FILE),
            directory.join(EXECUTION_FILE),
            directory.join(POSITION_FILE),
        );
        let (decisions, executions, positions) = match checkpoint {
            Some(value) => (
                TrackedFile::resume(paths.0, &value.decision)?,
                TrackedFile::resume(paths.1, &value.execution)?,
                TrackedFile::resume(paths.2, &value.position)?,
            ),
            None => {
                if paths.0.exists() || paths.1.exists() || paths.2.exists() {
                    return Err(EvaluationError(
                        "stream output ledger already exists; choose a new output directory".into(),
                    ));
                }
                (
                    TrackedFile::create(paths.0)?,
                    TrackedFile::create(paths.1)?,
                    TrackedFile::create(paths.2)?,
                )
            }
        };
        Ok(Self {
            decisions,
            executions,
            positions,
        })
    }

    fn flush(&mut self) -> Result<()> {
        self.decisions.flush()?;
        self.executions.flush()?;
        self.positions.flush()
    }

    fn checkpoint(&self) -> OutputCheckpoint {
        OutputCheckpoint {
            decision: self.decisions.state(),
            execution: self.executions.state(),
            position: self.positions.state(),
        }
    }
}

struct ReplayState {
    days: Vec<DayAccumulator>,
    units: i64,
    ending_target_units: i64,
    completed_directional_trades: usize,
    total_turnover_units: u64,
    total_cost_units: i64,
    processed_intervals: usize,
    market_point_count: usize,
    consumed_decisions: usize,
    last_decision_time: Option<EventTime>,
    last_execution_time: Option<EventTime>,
    last_market: MarketPoint,
}

impl ReplayState {
    fn new(config: &EvaluationConfig, first_market: MarketPoint) -> Self {
        Self {
            days: vec![
                DayAccumulator {
                    gross_factor: 1.0,
                    net_factor: 1.0,
                    cost: 0,
                    turnover: 0
                };
                config.evaluation_days.len()
            ],
            units: 0,
            ending_target_units: 0,
            completed_directional_trades: 0,
            total_turnover_units: 0,
            total_cost_units: 0,
            processed_intervals: 0,
            market_point_count: 1,
            consumed_decisions: 0,
            last_decision_time: None,
            last_execution_time: None,
            last_market: first_market,
        }
    }

    fn from_checkpoint(checkpoint: StreamCheckpoint) -> Self {
        Self {
            days: checkpoint
                .days
                .into_iter()
                .map(DayCheckpoint::into_day)
                .collect(),
            units: checkpoint.units,
            ending_target_units: checkpoint.ending_target_units,
            completed_directional_trades: checkpoint.completed_directional_trades,
            total_turnover_units: checkpoint.total_turnover_units,
            total_cost_units: checkpoint.total_cost_units,
            processed_intervals: checkpoint.processed_intervals,
            market_point_count: checkpoint.market_point_count,
            consumed_decisions: checkpoint.consumed_decisions,
            last_decision_time: checkpoint.last_decision_time,
            last_execution_time: checkpoint.last_execution_time,
            last_market: checkpoint.last_market.to_point(),
        }
    }

    fn checkpoint(
        &self,
        config_sha256: &str,
        market_sha256: &str,
        decision_sha256: &str,
        outputs: OutputCheckpoint,
        completed: bool,
    ) -> StreamCheckpoint {
        StreamCheckpoint {
            schema_version: SCHEMA_VERSION,
            config_sha256: config_sha256.to_owned(),
            market_sha256: market_sha256.to_owned(),
            decision_sha256: decision_sha256.to_owned(),
            processed_intervals: self.processed_intervals,
            market_point_count: self.market_point_count,
            consumed_decisions: self.consumed_decisions,
            units: self.units,
            ending_target_units: self.ending_target_units,
            completed_directional_trades: self.completed_directional_trades,
            total_turnover_units: self.total_turnover_units,
            total_cost_units: self.total_cost_units,
            last_decision_time: self.last_decision_time,
            last_execution_time: self.last_execution_time,
            last_market: MarketCheckpoint::from_point(&self.last_market),
            days: self.days.iter().map(DayCheckpoint::from).collect(),
            outputs,
            completed,
        }
    }
}

pub fn evaluate_jsonl(options: JsonlReplayOptions) -> Result<StreamRunSummary> {
    validate_options(&options)?;
    crate::validation::validate_config(&options.config)?;
    let config_sha256 = audit::hash(&options.config)?;
    let checkpoint_path = options.output_dir.join(CHECKPOINT_FILE);
    let checkpoint = if options.resume {
        Some(read_checkpoint(&checkpoint_path)?)
    } else {
        None
    };
    if let Some(value) = &checkpoint {
        require_stream(!value.completed, "checkpoint is already complete")?;
        require_stream(
            value.schema_version == SCHEMA_VERSION,
            "unsupported checkpoint schema version",
        )?;
        require_stream(
            value.config_sha256 == config_sha256
                && value.market_sha256 == options.market_sha256
                && value.decision_sha256 == options.decision_sha256,
            "checkpoint provenance differs from input",
        )?;
    } else if options.resume {
        return Err(EvaluationError("--resume requires checkpoint.json".into()));
    }
    if let Some(limit) = options.stop_after_intervals
        && checkpoint
            .as_ref()
            .is_some_and(|value| limit <= value.processed_intervals)
    {
        return Err(EvaluationError(
            "stop_after_intervals is not after checkpoint progress".into(),
        ));
    }
    let mut output = LedgerFiles::new(
        &options.output_dir,
        checkpoint.as_ref().map(|value| &value.outputs),
    )?;
    let mut market = JsonlReader::<MarketPoint>::open(&options.market_path)?;
    let mut decisions = JsonlReader::<TargetDecision>::open(&options.decisions_path)?;
    let mut state = match checkpoint {
        Some(value) => {
            market.skip(value.processed_intervals)?;
            let actual = market
                .next()?
                .ok_or_else(|| EvaluationError("market ended before checkpoint".into()))?;
            require_stream(
                actual == value.last_market.to_point(),
                "market prefix differs from checkpoint",
            )?;
            decisions.skip(value.consumed_decisions)?;
            ReplayState::from_checkpoint(value)
        }
        None => {
            let first = market
                .next()?
                .ok_or_else(|| EvaluationError("market stream is empty".into()))?;
            validate_market(&options.config, None, &first)?;
            ReplayState::new(&options.config, first)
        }
    };
    let mut pending_decision = None;
    while let Some(end) = market.next()? {
        validate_market(&options.config, Some(&state.last_market), &end)?;
        let terminal = market.is_empty()?;
        if pending_decision.is_none() {
            pending_decision = next_decision(&mut decisions, &options.config, &state)?;
        }
        if pending_decision
            .as_ref()
            .is_some_and(|value: &TargetDecision| value.execution_time < state.last_market.time)
        {
            return Err(EvaluationError(
                "decision execution time was skipped by market".into(),
            ));
        }
        let decision = if pending_decision
            .as_ref()
            .is_some_and(|value| value.execution_time == state.last_market.time)
        {
            pending_decision.take()
        } else {
            None
        };
        let start = state.last_market.clone();
        process_interval(
            &options.config,
            &mut state,
            &mut output,
            &start,
            &end,
            decision,
            terminal,
        )?;
        state.last_market = end;
        let stopped = options
            .stop_after_intervals
            .is_some_and(|limit| state.processed_intervals >= limit && !terminal);
        let periodic =
            state.processed_intervals % options.checkpoint_every_intervals == 0 && !terminal;
        if stopped || periodic {
            output.flush()?;
            write_checkpoint(
                &checkpoint_path,
                &state.checkpoint(
                    &config_sha256,
                    &options.market_sha256,
                    &options.decision_sha256,
                    output.checkpoint(),
                    false,
                ),
            )?;
        }
        if stopped {
            return Ok(summary_checkpoint(
                &options.output_dir,
                &state,
                output.executions.lines,
            ));
        }
    }
    if pending_decision.is_some() || decisions.next()?.is_some() {
        return Err(EvaluationError(
            "decision stream contains an execution without a market interval".into(),
        ));
    }
    if state.processed_intervals == 0 {
        return Err(EvaluationError("market stream has no interval".into()));
    }
    let daily = build_daily_returns(&state.days, &options.config)?;
    write_daily(&options.output_dir.join(DAILY_FILE), &daily)?;
    let mut metrics = build_metrics(&state.days, &options.config)?;
    metrics.turnover_units = state.total_turnover_units;
    metrics.total_cost_units = state.total_cost_units;
    metrics.completed_directional_trades = state.completed_directional_trades;
    metrics.ending_target_units = state.ending_target_units;
    metrics.ending_realized_units = state.units;
    output.flush()?;
    let ledger_sha256 = hash_ledger(&options.output_dir, &daily, &metrics)?;
    let run_sha256 = audit::hash(&(
        SCHEMA_VERSION,
        &config_sha256,
        &options.market_sha256,
        &options.decision_sha256,
        &ledger_sha256,
    ))?;
    let audit_manifest = crate::AuditManifest {
        schema_version: SCHEMA_VERSION,
        market_point_count: state.market_point_count,
        decision_count: state.consumed_decisions,
        execution_count: output.executions.lines,
        interval_count: state.processed_intervals,
        config_sha256,
        market_sha256: options.market_sha256,
        decision_sha256: options.decision_sha256,
        ledger_sha256,
        run_sha256,
    };
    write_checkpoint(
        &checkpoint_path,
        &state.checkpoint(
            &audit_manifest.config_sha256,
            &audit_manifest.market_sha256,
            &audit_manifest.decision_sha256,
            output.checkpoint(),
            true,
        ),
    )?;
    Ok(StreamRunSummary {
        schema_version: SCHEMA_VERSION,
        status: StreamStatus::Complete,
        processed_intervals: state.processed_intervals,
        market_point_count: state.market_point_count,
        decision_count: state.consumed_decisions,
        execution_count: output.executions.lines,
        output: manifest(&options.output_dir),
        metrics: Some(metrics),
        audit: Some(audit_manifest),
    })
}

fn process_interval(
    config: &EvaluationConfig,
    state: &mut ReplayState,
    output: &mut LedgerFiles,
    start: &MarketPoint,
    end: &MarketPoint,
    decision: Option<TargetDecision>,
    terminal: bool,
) -> Result<()> {
    let mut turnover = 0_u64;
    if let Some(decision) = decision {
        let delta = decision
            .target_units
            .checked_sub(state.units)
            .ok_or_else(|| EvaluationError("target delta overflows".into()))?;
        turnover = delta.unsigned_abs();
        if delta != 0 {
            output.executions.write(&execution(
                start,
                state.units,
                decision.target_units,
                Some(state.consumed_decisions),
                ExecutionReason::Rebalance,
                config,
            )?)?;
        }
        if state.units != 0 && state.units.signum() != decision.target_units.signum() {
            state.completed_directional_trades += 1;
        }
        state.last_decision_time = Some(decision.decision_time);
        state.last_execution_time = Some(decision.execution_time);
        state.units = decision.target_units;
        state.ending_target_units = state.units;
        let mut canonical = decision;
        canonical.diagnostic_json.sort_all_objects();
        output.decisions.write(&DecisionLedgerRow {
            decision: canonical,
            realized_units: state.units,
        })?;
        state.consumed_decisions += 1;
    }
    let liquidation =
        terminal && config.terminal_policy == TerminalPolicy::Liquidate && state.units != 0;
    if liquidation {
        turnover = add_turnover(turnover, state.units.unsigned_abs())?;
        output.executions.write(&execution(
            end,
            state.units,
            0,
            None,
            ExecutionReason::TerminalLiquidation,
            config,
        )?)?;
        state.completed_directional_trades += 1;
    }
    let gross = state.units as f64 * (end.price / start.price - 1.0);
    let cost = turnover as f64 * (config.one_way_cost_bps + config.slippage_bps) / 10_000.0;
    let net = gross - cost;
    require_stream(
        gross.is_finite() && net.is_finite() && net > -1.0,
        format!(
            "interval {} has non-finite return or exhausted equity",
            state.processed_intervals
        ),
    )?;
    let cost_units = to_fixed(cost)?;
    let day = end.time.timestamp_ns / NANOS_PER_DAY * NANOS_PER_DAY;
    let day_index = config
        .evaluation_days
        .binary_search(&day)
        .map_err(|_| EvaluationError("unregistered realization day".into()))?;
    state.days[day_index].gross_factor *= 1.0 + gross;
    state.days[day_index].net_factor *= 1.0 + net;
    state.days[day_index].cost = add(state.days[day_index].cost, cost_units)?;
    state.days[day_index].turnover = add_turnover(state.days[day_index].turnover, turnover)?;
    state.total_turnover_units = add_turnover(state.total_turnover_units, turnover)?;
    state.total_cost_units = add(state.total_cost_units, cost_units)?;
    output.positions.write(&PositionLedgerRow {
        start_time: start.time,
        end_time: end.time,
        instrument: config.instrument.clone(),
        realized_units: state.units,
        turnover_units: turnover,
        gross_return_units: to_fixed(gross)?,
        cost_units,
        net_return_units: to_fixed(net)?,
    })?;
    if liquidation {
        state.units = 0;
    }
    state.processed_intervals += 1;
    state.market_point_count += 1;
    Ok(())
}

fn next_decision(
    reader: &mut JsonlReader<TargetDecision>,
    config: &EvaluationConfig,
    state: &ReplayState,
) -> Result<Option<TargetDecision>> {
    let Some(value) = reader.next()? else {
        return Ok(None);
    };
    validate_decision(config, state, &value)?;
    Ok(Some(value))
}

fn validate_decision(
    config: &EvaluationConfig,
    state: &ReplayState,
    value: &TargetDecision,
) -> Result<()> {
    require_stream(
        value.instrument == config.instrument,
        "decision instrument differs from config",
    )?;
    require_stream(
        value.target_units.unsigned_abs() <= (1_u64 << 53),
        "target exceeds exact binary64 integer range",
    )?;
    require_stream(
        value.provenance_hash.len() == 64
            && value
                .provenance_hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "provenance_hash must be lowercase SHA-256 hex",
    )?;
    let CausalAvailability {
        source_end,
        available_at,
    } = &value.causal_availability;
    require_stream(
        *source_end <= *available_at
            && *available_at <= value.decision_time
            && value.decision_time < value.execution_time,
        "source/availability/decision/fill ordering is non-causal",
    )?;
    require_stream(
        state
            .last_decision_time
            .is_none_or(|time| time <= value.decision_time),
        "decision times are not ordered",
    )?;
    require_stream(
        state
            .last_execution_time
            .is_none_or(|time| time < value.execution_time),
        "execution keys must be unique and strictly ordered",
    )
}

fn validate_market(
    config: &EvaluationConfig,
    previous: Option<&MarketPoint>,
    value: &MarketPoint,
) -> Result<()> {
    require_stream(
        value.instrument == config.instrument,
        "market instrument differs from config",
    )?;
    require_stream(
        value.price.is_finite() && value.price > 0.0,
        "market prices must be finite and positive",
    )?;
    let day = value.time.timestamp_ns / NANOS_PER_DAY * NANOS_PER_DAY;
    require_stream(
        config.evaluation_days.binary_search(&day).is_ok(),
        "unregistered realization day",
    )?;
    if let Some(previous) = previous {
        require_stream(
            previous.time.timestamp_ns < value.time.timestamp_ns,
            "market timestamps must be strictly increasing",
        )?;
    }
    Ok(())
}

fn validate_options(options: &JsonlReplayOptions) -> Result<()> {
    require_stream(
        options.checkpoint_every_intervals > 0,
        "checkpoint_every_intervals must be positive",
    )?;
    require_stream(
        options.market_sha256.len() == 64 && options.decision_sha256.len() == 64,
        "input SHA-256 values must be 64 hex characters",
    )?;
    if let Some(value) = options.stop_after_intervals {
        require_stream(value > 0, "stop_after_intervals must be positive")?;
    }
    Ok(())
}

fn write_checkpoint(path: &Path, checkpoint: &StreamCheckpoint) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(checkpoint)
        .map_err(|error| EvaluationError(format!("cannot serialize checkpoint: {error}")))?;
    fs::write(&temporary, bytes).map_err(|error| io_error(&temporary, error))?;
    fs::rename(&temporary, path).map_err(|error| io_error(path, error))
}

fn read_checkpoint(path: &Path) -> Result<StreamCheckpoint> {
    let file = File::open(path).map_err(|error| io_error(path, error))?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|error| EvaluationError(format!("cannot parse checkpoint: {error}")))
}

fn write_daily(path: &Path, rows: &[DailyReturnRow]) -> Result<()> {
    if path.exists() {
        return Err(EvaluationError(format!(
            "daily ledger already exists: {}",
            path.display()
        )));
    }
    let mut writer = BufWriter::new(File::create_new(path).map_err(|error| io_error(path, error))?);
    for row in rows {
        let bytes = serde_json::to_vec(row)
            .map_err(|error| EvaluationError(format!("cannot serialize daily row: {error}")))?;
        writer
            .write_all(&bytes)
            .and_then(|_| writer.write_all(b"\n"))
            .map_err(|error| io_error(path, error))?;
    }
    writer.flush().map_err(|error| io_error(path, error))
}

fn hash_ledger(
    directory: &Path,
    daily: &[DailyReturnRow],
    metrics: &EvaluationMetrics,
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"[");
    hash_file_array(&mut hasher, &directory.join(DECISION_FILE))?;
    hasher.update(b",");
    hash_file_array(&mut hasher, &directory.join(EXECUTION_FILE))?;
    hasher.update(b",");
    hash_file_array(&mut hasher, &directory.join(POSITION_FILE))?;
    hasher.update(b",");
    hash_slice_array(&mut hasher, daily)?;
    hasher.update(b",");
    hasher.update(
        serde_json::to_vec(metrics)
            .map_err(|error| EvaluationError(format!("cannot serialize metrics: {error}")))?,
    );
    hasher.update(b"]");
    Ok(hex(&hasher.finalize()))
}

fn hash_file_array(hasher: &mut Sha256, path: &Path) -> Result<()> {
    let reader = BufReader::new(File::open(path).map_err(|error| io_error(path, error))?);
    hasher.update(b"[");
    let mut first = true;
    for line in reader.lines() {
        let line = line.map_err(|error| io_error(path, error))?;
        if !first {
            hasher.update(b",");
        }
        hasher.update(line.as_bytes());
        first = false;
    }
    hasher.update(b"]");
    Ok(())
}

fn hash_slice_array<T: Serialize>(hasher: &mut Sha256, values: &[T]) -> Result<()> {
    hasher.update(b"[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            hasher.update(b",");
        }
        hasher.update(
            serde_json::to_vec(value).map_err(|error| {
                EvaluationError(format!("cannot serialize JSON array: {error}"))
            })?,
        );
    }
    hasher.update(b"]");
    Ok(())
}

pub fn hash_market_jsonl(path: impl AsRef<Path>) -> Result<String> {
    hash_input_jsonl_market(path.as_ref())
}
pub fn hash_decision_jsonl(path: impl AsRef<Path>) -> Result<String> {
    hash_input_jsonl_decision(path.as_ref())
}

fn hash_input_jsonl_market(path: &Path) -> Result<String> {
    let reader = BufReader::new(File::open(path).map_err(|error| io_error(path, error))?);
    let mut hasher = Sha256::new();
    hasher.update(b"[");
    let mut first = true;
    for line in reader.lines() {
        let line = line.map_err(|error| io_error(path, error))?;
        let value: MarketPoint = serde_json::from_str(&line).map_err(|error| {
            EvaluationError(format!("cannot parse {}: {error}", path.display()))
        })?;
        if !first {
            hasher.update(b",");
        }
        hasher.update(serde_json::to_vec(&value).map_err(|error| {
            EvaluationError(format!("cannot serialize {}: {error}", path.display()))
        })?);
        first = false;
    }
    hasher.update(b"]");
    Ok(hex(&hasher.finalize()))
}

fn hash_input_jsonl_decision(path: &Path) -> Result<String> {
    let reader = BufReader::new(File::open(path).map_err(|error| io_error(path, error))?);
    let mut hasher = Sha256::new();
    hasher.update(b"[");
    let mut first = true;
    for line in reader.lines() {
        let line = line.map_err(|error| io_error(path, error))?;
        let mut value: TargetDecision = serde_json::from_str(&line).map_err(|error| {
            EvaluationError(format!("cannot parse {}: {error}", path.display()))
        })?;
        value.diagnostic_json.sort_all_objects();
        if !first {
            hasher.update(b",");
        }
        hasher.update(serde_json::to_vec(&value).map_err(|error| {
            EvaluationError(format!("cannot serialize {}: {error}", path.display()))
        })?);
        first = false;
    }
    hasher.update(b"]");
    Ok(hex(&hasher.finalize()))
}

fn file_state(path: &Path) -> Result<OutputFileState> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|error| io_error(path, error))?
        .read_to_end(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    Ok(OutputFileState {
        bytes: bytes.len() as u64,
        lines: bytes.iter().filter(|byte| **byte == b'\n').count(),
        sha256: hex(&Sha256::digest(&bytes)),
    })
}

fn manifest(directory: &Path) -> StreamOutputManifest {
    StreamOutputManifest {
        decision_ledger: directory.join(DECISION_FILE).display().to_string(),
        execution_ledger: directory.join(EXECUTION_FILE).display().to_string(),
        position_ledger: directory.join(POSITION_FILE).display().to_string(),
        daily_returns: directory.join(DAILY_FILE).display().to_string(),
        checkpoint: directory.join(CHECKPOINT_FILE).display().to_string(),
        summary: directory.join(SUMMARY_FILE).display().to_string(),
    }
}

fn summary_checkpoint(
    directory: &Path,
    state: &ReplayState,
    execution_count: usize,
) -> StreamRunSummary {
    StreamRunSummary {
        schema_version: SCHEMA_VERSION,
        status: StreamStatus::CheckpointOnly,
        processed_intervals: state.processed_intervals,
        market_point_count: state.market_point_count,
        decision_count: state.consumed_decisions,
        execution_count,
        output: manifest(directory),
        metrics: None,
        audit: None,
    }
}

fn require_stream(valid: bool, message: impl Into<String>) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(EvaluationError(message.into()))
    }
}
fn io_error(path: &Path, error: std::io::Error) -> EvaluationError {
    EvaluationError(format!("{}: {error}", path.display()))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

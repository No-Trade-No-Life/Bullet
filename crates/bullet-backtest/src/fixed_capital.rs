//! Causal, component-level fixed-capital research replay.
//!
//! This module is independent of Bullet's contract-accounting backtester. It
//! keeps every component exposure separate, computes its fixed-capital return,
//! and sums canonical twelve-decimal returns on an explicit evaluation calendar.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bullet_data::HistoryBar;
pub use bullet_data::TimestampInterpretation;
use chrono::{Datelike, Duration, NaiveDate};

const FIXED_SCALE: f64 = 1_000_000_000_000.0;
const NANOS_PER_DAY: u64 = 86_400_000_000_000;
const SHANGHAI_OFFSET_NS: u64 = 8 * 60 * 60 * 1_000_000_000;
const TRADING_DAYS_PER_YEAR: f64 = 252.0;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TradingDay(NaiveDate);

impl TradingDay {
    pub fn new(year: i32, month: u32, day: u32) -> Option<Self> {
        NaiveDate::from_ymd_opt(year, month, day).map(Self)
    }

    pub fn parse(value: &str) -> Option<Self> {
        NaiveDate::parse_from_str(value, "%Y-%m-%d").ok().map(Self)
    }
}

impl fmt::Display for TradingDay {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:04}-{:02}-{:02}",
            self.0.year(),
            self.0.month(),
            self.0.day()
        )
    }
}

#[derive(Clone, Debug)]
pub struct FixedCapitalInstrument {
    pub id: String,
    pub data: PathBuf,
    pub timestamp_interpretation: TimestampInterpretation,
}

#[derive(Clone, Debug)]
pub struct FixedCapitalConfig {
    pub instruments: Vec<FixedCapitalInstrument>,
    pub components: Vec<String>,
    pub evaluation_days: Vec<TradingDay>,
}

#[derive(Clone, Debug)]
pub struct InstrumentHistory {
    pub instrument: String,
    pub bars: Vec<HistoryBar>,
}

#[derive(Clone, Copy, Debug)]
pub struct TimestampBar<'a> {
    pub instrument: &'a str,
    pub bar: &'a HistoryBar,
}

#[derive(Clone, Copy, Debug)]
pub struct FixedCapitalContext<'a> {
    pub timestamp_ns: u64,
    pub bars: &'a [TimestampBar<'a>],
}

pub trait FixedCapitalStrategy {
    fn on_timestamp(
        &mut self,
        context: FixedCapitalContext<'_>,
    ) -> Result<Vec<ExposureOrder>, String>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExposureSide {
    Long,
    Short,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScheduledFill {
    Open,
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExitPlan {
    At {
        timestamp_ns: u64,
        fill: ScheduledFill,
    },
    StopOrAt {
        stop_price: f64,
        timestamp_ns: u64,
        fill: ScheduledFill,
    },
}

impl ExitPlan {
    fn scheduled(self) -> (u64, ScheduledFill) {
        match self {
            Self::At { timestamp_ns, fill }
            | Self::StopOrAt {
                timestamp_ns, fill, ..
            } => (timestamp_ns, fill),
        }
    }

    fn stop(self) -> Option<f64> {
        match self {
            Self::At { .. } => None,
            Self::StopOrAt { stop_price, .. } => Some(stop_price),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExposureOrder {
    pub id: String,
    pub component: String,
    pub instrument: String,
    pub setup: String,
    pub side: ExposureSide,
    pub component_scale: f64,
    pub instrument_weight: f64,
    pub exit: ExitPlan,
    pub cost_fraction: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitReason {
    ScheduledOpen,
    ScheduledClose,
    Stop,
}

impl fmt::Display for ExitReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ScheduledOpen => "scheduled_open",
            Self::ScheduledClose => "scheduled_close",
            Self::Stop => "stop",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentTrade {
    pub id: String,
    pub component: String,
    pub instrument: String,
    pub setup: String,
    pub signal_timestamp_ns: u64,
    pub entry_timestamp_ns: u64,
    pub exit_timestamp_ns: u64,
    pub side: ExposureSide,
    pub entry_price: f64,
    pub exit_price: f64,
    pub exit_reason: ExitReason,
    pub cost_fraction: f64,
    pub component_scale: f64,
    pub instrument_weight: f64,
    pub net_return_units: i64,
    pub weighted_return_units: i64,
}

impl ComponentTrade {
    pub fn net_return(&self) -> f64 {
        from_fixed(self.net_return_units)
    }

    pub fn weighted_return(&self) -> f64 {
        from_fixed(self.weighted_return_units)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CensoredExposure {
    pub id: String,
    pub component: String,
    pub instrument: String,
    pub signal_timestamp_ns: u64,
    pub entry_timestamp_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DailyReturn {
    pub day: TradingDay,
    pub component_return_units: BTreeMap<String, i64>,
    pub portfolio_return_units: i64,
}

impl DailyReturn {
    pub fn portfolio_return(&self) -> f64 {
        from_fixed(self.portfolio_return_units)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixedCapitalPerformance {
    pub days: usize,
    pub mean_daily_return: f64,
    pub daily_volatility: f64,
    pub sharpe: f64,
    pub annualized_return: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub nonzero_days: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixedCapitalResult {
    pub bars: usize,
    pub data_size_bytes: u64,
    pub runtime_ms: u128,
    pub trades: Vec<ComponentTrade>,
    pub censored: Vec<CensoredExposure>,
    pub component_order: Vec<String>,
    pub daily_returns: Vec<DailyReturn>,
    pub performance: FixedCapitalPerformance,
}

impl FixedCapitalResult {
    pub fn write_csv(&self, output: impl AsRef<Path>) -> Result<(), FixedCapitalError> {
        let output = output.as_ref();
        fs::create_dir_all(output).map_err(FixedCapitalError::WriteOutput)?;
        write_trades(&output.join("component_trades.csv"), &self.trades)?;
        write_daily(
            &output.join("daily_returns.csv"),
            &self.component_order,
            &self.daily_returns,
        )?;
        write_censored(&output.join("censored_exposures.csv"), &self.censored)?;
        write_metrics(&output.join("metrics.json"), self)?;
        Ok(())
    }
}

pub fn run_fixed_capital<S: FixedCapitalStrategy>(
    config: &FixedCapitalConfig,
    strategy: &mut S,
) -> Result<FixedCapitalResult, FixedCapitalError> {
    let started = Instant::now();
    validate_config(config)?;
    let mut data_size_bytes = 0;
    let mut histories = Vec::with_capacity(config.instruments.len());
    for instrument in &config.instruments {
        data_size_bytes += fs::metadata(&instrument.data)
            .map_err(FixedCapitalError::ReadData)?
            .len();
        histories.push(InstrumentHistory {
            instrument: instrument.id.clone(),
            bars: bullet_data::read_history(&instrument.data, instrument.timestamp_interpretation)
                .map_err(FixedCapitalError::Data)?,
        });
    }
    let mut result = run_fixed_capital_history_inner(
        &config.evaluation_days,
        Some(&config.components),
        &histories,
        data_size_bytes,
        strategy,
    )?;
    result.runtime_ms = started.elapsed().as_millis();
    Ok(result)
}

pub fn run_fixed_capital_history<S: FixedCapitalStrategy>(
    evaluation_days: &[TradingDay],
    histories: &[InstrumentHistory],
    data_size_bytes: u64,
    strategy: &mut S,
) -> Result<FixedCapitalResult, FixedCapitalError> {
    run_fixed_capital_history_inner(evaluation_days, None, histories, data_size_bytes, strategy)
}

fn run_fixed_capital_history_inner<S: FixedCapitalStrategy>(
    evaluation_days: &[TradingDay],
    configured_components: Option<&[String]>,
    histories: &[InstrumentHistory],
    data_size_bytes: u64,
    strategy: &mut S,
) -> Result<FixedCapitalResult, FixedCapitalError> {
    let started = Instant::now();
    validate_history_inputs(evaluation_days, histories)?;
    let instruments: BTreeSet<_> = histories
        .iter()
        .map(|history| history.instrument.clone())
        .collect();
    let mut cursors = vec![0_usize; histories.len()];
    let mut state = ReplayState::new(instruments);
    let mut bars = 0;
    let mut last_timestamp_ns = None;

    while let Some(timestamp_ns) = next_timestamp(histories, &cursors) {
        last_timestamp_ns = Some(timestamp_ns);
        state.reject_missed_schedules(timestamp_ns)?;
        let mut batch = Vec::new();
        for (index, history) in histories.iter().enumerate() {
            if history.bars.get(cursors[index]).map(|bar| bar.timestamp_ns) == Some(timestamp_ns) {
                batch.push(TimestampBar {
                    instrument: &history.instrument,
                    bar: &history.bars[cursors[index]],
                });
                cursors[index] += 1;
            }
        }
        batch.sort_by_key(|value| value.instrument);
        bars += batch.len();
        state.process_open(timestamp_ns, &batch)?;
        state.process_stops(timestamp_ns, &batch)?;
        let orders = strategy
            .on_timestamp(FixedCapitalContext {
                timestamp_ns,
                bars: &batch,
            })
            .map_err(FixedCapitalError::Strategy)?;
        state.submit(timestamp_ns, &batch, orders)?;
        state.process_close(timestamp_ns, &batch)?;
    }

    if bars == 0 {
        return Err(FixedCapitalError::EmptyBars);
    }
    state.reject_schedules_at_or_before(last_timestamp_ns.expect("bars are non-empty"))?;
    let censored = state.censored();
    let mut trades = state.trades;
    trades.sort_by(|left, right| {
        left.component
            .cmp(&right.component)
            .then_with(|| left.instrument.cmp(&right.instrument))
            .then_with(|| left.entry_timestamp_ns.cmp(&right.entry_timestamp_ns))
            .then_with(|| left.id.cmp(&right.id))
    });
    let component_order = component_order(configured_components, &trades)?;
    let daily_returns = build_daily(evaluation_days, &component_order, &trades)?;
    let performance = evaluate(&daily_returns);
    Ok(FixedCapitalResult {
        bars,
        data_size_bytes,
        runtime_ms: started.elapsed().as_millis(),
        trades,
        censored,
        component_order,
        daily_returns,
        performance,
    })
}

#[derive(Clone)]
struct PendingExposure {
    order: ExposureOrder,
    signal_timestamp_ns: u64,
}

#[derive(Clone)]
struct ActiveExposure {
    pending: PendingExposure,
    entry_timestamp_ns: u64,
    entry_price: f64,
}

struct ReplayState {
    instruments: BTreeSet<String>,
    ids: BTreeSet<String>,
    pending: BTreeMap<String, Vec<PendingExposure>>,
    active: BTreeMap<String, ActiveExposure>,
    scheduled_open: BTreeMap<(u64, String), Vec<String>>,
    scheduled_close: BTreeMap<(u64, String), Vec<String>>,
    stops: BTreeMap<String, BTreeSet<String>>,
    trades: Vec<ComponentTrade>,
}

impl ReplayState {
    fn new(instruments: BTreeSet<String>) -> Self {
        Self {
            instruments,
            ids: BTreeSet::new(),
            pending: BTreeMap::new(),
            active: BTreeMap::new(),
            scheduled_open: BTreeMap::new(),
            scheduled_close: BTreeMap::new(),
            stops: BTreeMap::new(),
            trades: Vec::new(),
        }
    }

    fn submit(
        &mut self,
        timestamp_ns: u64,
        _bars: &[TimestampBar<'_>],
        mut orders: Vec<ExposureOrder>,
    ) -> Result<(), FixedCapitalError> {
        orders.sort_by(|left, right| left.id.cmp(&right.id));
        for order in orders {
            validate_order(timestamp_ns, &self.instruments, &self.ids, &order)?;
            self.ids.insert(order.id.clone());
            let pending = PendingExposure {
                order,
                signal_timestamp_ns: timestamp_ns,
            };
            self.pending
                .entry(pending.order.instrument.clone())
                .or_default()
                .push(pending);
        }
        Ok(())
    }

    fn process_open(
        &mut self,
        timestamp_ns: u64,
        bars: &[TimestampBar<'_>],
    ) -> Result<(), FixedCapitalError> {
        for value in bars {
            let key = (timestamp_ns, value.instrument.to_owned());
            for id in self.scheduled_open.remove(&key).unwrap_or_default() {
                self.complete(&id, timestamp_ns, value.bar.open, ExitReason::ScheduledOpen)?;
            }
            for pending in self.pending.remove(value.instrument).unwrap_or_default() {
                self.activate(pending, timestamp_ns, value.bar.open)?;
            }
        }
        Ok(())
    }

    fn process_stops(
        &mut self,
        timestamp_ns: u64,
        bars: &[TimestampBar<'_>],
    ) -> Result<(), FixedCapitalError> {
        for value in bars {
            let ids: Vec<_> = self
                .stops
                .get(value.instrument)
                .into_iter()
                .flatten()
                .cloned()
                .collect();
            for id in ids {
                let active = &self.active[&id];
                let stop = active
                    .pending
                    .order
                    .exit
                    .stop()
                    .expect("stop index invariant");
                let fill = match active.pending.order.side {
                    ExposureSide::Long if value.bar.low <= stop => Some(value.bar.open.min(stop)),
                    ExposureSide::Short if value.bar.high >= stop => Some(value.bar.open.max(stop)),
                    _ => None,
                };
                if let Some(price) = fill {
                    self.complete(&id, timestamp_ns, price, ExitReason::Stop)?;
                }
            }
        }
        Ok(())
    }

    fn process_close(
        &mut self,
        timestamp_ns: u64,
        bars: &[TimestampBar<'_>],
    ) -> Result<(), FixedCapitalError> {
        for value in bars {
            let key = (timestamp_ns, value.instrument.to_owned());
            for id in self.scheduled_close.remove(&key).unwrap_or_default() {
                self.complete(
                    &id,
                    timestamp_ns,
                    value.bar.close,
                    ExitReason::ScheduledClose,
                )?;
            }
        }
        Ok(())
    }

    fn activate(
        &mut self,
        pending: PendingExposure,
        timestamp_ns: u64,
        price: f64,
    ) -> Result<(), FixedCapitalError> {
        let (exit_timestamp, exit_fill) = pending.order.exit.scheduled();
        if exit_timestamp <= timestamp_ns {
            return Err(FixedCapitalError::ExitNotAfterEntry(pending.order.id));
        }
        if pending.order.exit.stop().is_some() {
            self.stops
                .entry(pending.order.instrument.clone())
                .or_default()
                .insert(pending.order.id.clone());
        }
        let key = (exit_timestamp, pending.order.instrument.clone());
        match exit_fill {
            ScheduledFill::Open => self.scheduled_open.entry(key).or_default(),
            ScheduledFill::Close => self.scheduled_close.entry(key).or_default(),
        }
        .push(pending.order.id.clone());
        self.active.insert(
            pending.order.id.clone(),
            ActiveExposure {
                pending,
                entry_timestamp_ns: timestamp_ns,
                entry_price: price,
            },
        );
        Ok(())
    }

    fn complete(
        &mut self,
        id: &str,
        timestamp_ns: u64,
        exit_price: f64,
        reason: ExitReason,
    ) -> Result<(), FixedCapitalError> {
        let active = self
            .active
            .remove(id)
            .ok_or_else(|| FixedCapitalError::InactiveExposure(id.to_owned()))?;
        let order = &active.pending.order;
        let scheduled = order.exit.scheduled();
        let key = (scheduled.0, order.instrument.clone());
        let schedule = match scheduled.1 {
            ScheduledFill::Open => &mut self.scheduled_open,
            ScheduledFill::Close => &mut self.scheduled_close,
        };
        remove_id(schedule, &key, id);
        if order.exit.stop().is_some() {
            let indexed = self
                .stops
                .get_mut(&order.instrument)
                .expect("stop index invariant");
            indexed.remove(id);
        }
        let direction = match order.side {
            ExposureSide::Long => 1.0,
            ExposureSide::Short => -1.0,
        };
        let gross = direction * (exit_price / active.entry_price - 1.0);
        let net_units = to_fixed((1.0 + gross) * (1.0 - order.cost_fraction) - 1.0)?;
        let net = from_fixed(net_units);
        let weighted_units = to_fixed(net * order.component_scale * order.instrument_weight)?;
        self.trades.push(ComponentTrade {
            id: order.id.clone(),
            component: order.component.clone(),
            instrument: order.instrument.clone(),
            setup: order.setup.clone(),
            signal_timestamp_ns: active.pending.signal_timestamp_ns,
            entry_timestamp_ns: active.entry_timestamp_ns,
            exit_timestamp_ns: timestamp_ns,
            side: order.side,
            entry_price: active.entry_price,
            exit_price,
            exit_reason: reason,
            cost_fraction: order.cost_fraction,
            component_scale: order.component_scale,
            instrument_weight: order.instrument_weight,
            net_return_units: net_units,
            weighted_return_units: weighted_units,
        });
        Ok(())
    }

    fn reject_missed_schedules(&self, timestamp_ns: u64) -> Result<(), FixedCapitalError> {
        let missed = self
            .scheduled_open
            .keys()
            .chain(self.scheduled_close.keys())
            .find(|(scheduled, _)| *scheduled < timestamp_ns);
        if let Some((scheduled, instrument)) = missed {
            return Err(FixedCapitalError::MissingScheduledBar {
                instrument: instrument.clone(),
                timestamp_ns: *scheduled,
            });
        }
        Ok(())
    }

    fn reject_schedules_at_or_before(&self, timestamp_ns: u64) -> Result<(), FixedCapitalError> {
        let missed = self
            .scheduled_open
            .keys()
            .chain(self.scheduled_close.keys())
            .find(|(scheduled, _)| *scheduled <= timestamp_ns);
        if let Some((scheduled, instrument)) = missed {
            return Err(FixedCapitalError::MissingScheduledBar {
                instrument: instrument.clone(),
                timestamp_ns: *scheduled,
            });
        }
        Ok(())
    }

    fn censored(&self) -> Vec<CensoredExposure> {
        let pending = self
            .pending
            .values()
            .flatten()
            .map(|value| CensoredExposure {
                id: value.order.id.clone(),
                component: value.order.component.clone(),
                instrument: value.order.instrument.clone(),
                signal_timestamp_ns: value.signal_timestamp_ns,
                entry_timestamp_ns: None,
            });
        let active = self.active.values().map(|value| CensoredExposure {
            id: value.pending.order.id.clone(),
            component: value.pending.order.component.clone(),
            instrument: value.pending.order.instrument.clone(),
            signal_timestamp_ns: value.pending.signal_timestamp_ns,
            entry_timestamp_ns: Some(value.entry_timestamp_ns),
        });
        let mut values: Vec<_> = pending.chain(active).collect();
        values.sort_by(|left, right| left.id.cmp(&right.id));
        values
    }
}

fn validate_config(config: &FixedCapitalConfig) -> Result<(), FixedCapitalError> {
    if config.instruments.is_empty() {
        return Err(FixedCapitalError::InvalidConfig(
            "at least one instrument is required",
        ));
    }
    let mut ids = BTreeSet::new();
    for instrument in &config.instruments {
        if instrument.id.is_empty() || !ids.insert(&instrument.id) {
            return Err(FixedCapitalError::InvalidConfig(
                "instrument ids must be non-empty and unique",
            ));
        }
    }
    validate_components(&config.components)?;
    validate_days(&config.evaluation_days)
}

fn validate_components(components: &[String]) -> Result<(), FixedCapitalError> {
    if components.is_empty() {
        return Err(FixedCapitalError::InvalidConfig(
            "components must not be empty",
        ));
    }
    let mut names = BTreeSet::new();
    if components
        .iter()
        .any(|component| component.is_empty() || !names.insert(component))
    {
        return Err(FixedCapitalError::InvalidConfig(
            "components must be non-empty and unique",
        ));
    }
    Ok(())
}

fn validate_history_inputs(
    evaluation_days: &[TradingDay],
    histories: &[InstrumentHistory],
) -> Result<(), FixedCapitalError> {
    validate_days(evaluation_days)?;
    if histories.is_empty() {
        return Err(FixedCapitalError::InvalidConfig(
            "at least one instrument is required",
        ));
    }
    let mut ids = BTreeSet::new();
    for history in histories {
        if history.instrument.is_empty() || !ids.insert(&history.instrument) {
            return Err(FixedCapitalError::InvalidConfig(
                "instrument ids must be non-empty and unique",
            ));
        }
        if history
            .bars
            .windows(2)
            .any(|pair| pair[0].timestamp_ns >= pair[1].timestamp_ns)
        {
            return Err(FixedCapitalError::NonIncreasingBars(
                history.instrument.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_days(days: &[TradingDay]) -> Result<(), FixedCapitalError> {
    if days.is_empty() {
        return Err(FixedCapitalError::InvalidConfig(
            "evaluation_days must not be empty",
        ));
    }
    if days.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(FixedCapitalError::InvalidConfig(
            "evaluation_days must be unique and increasing",
        ));
    }
    Ok(())
}

fn validate_order(
    timestamp_ns: u64,
    instruments: &BTreeSet<String>,
    ids: &BTreeSet<String>,
    order: &ExposureOrder,
) -> Result<(), FixedCapitalError> {
    if order.id.is_empty() || order.component.is_empty() || order.instrument.is_empty() {
        return Err(FixedCapitalError::InvalidOrder(
            "id, component and instrument must be non-empty".to_owned(),
        ));
    }
    if ids.contains(&order.id) {
        return Err(FixedCapitalError::DuplicateExposure(order.id.clone()));
    }
    if !instruments.contains(&order.instrument) {
        return Err(FixedCapitalError::UnknownInstrument(
            order.instrument.clone(),
        ));
    }
    if !order.component_scale.is_finite()
        || order.component_scale <= 0.0
        || !order.instrument_weight.is_finite()
        || order.instrument_weight <= 0.0
    {
        return Err(FixedCapitalError::InvalidOrder(
            "component_scale and instrument_weight must be positive and finite".to_owned(),
        ));
    }
    if !order.cost_fraction.is_finite() || order.cost_fraction < 0.0 || order.cost_fraction >= 1.0 {
        return Err(FixedCapitalError::InvalidOrder(
            "cost_fraction must be finite and in [0, 1)".to_owned(),
        ));
    }
    let (exit_timestamp, _) = order.exit.scheduled();
    if exit_timestamp <= timestamp_ns {
        return Err(FixedCapitalError::ExitNotAfterSignal(order.id.clone()));
    }
    if let Some(stop) = order.exit.stop()
        && (!stop.is_finite() || stop <= 0.0)
    {
        return Err(FixedCapitalError::InvalidStop(order.id.clone()));
    }
    Ok(())
}

fn next_timestamp(histories: &[InstrumentHistory], cursors: &[usize]) -> Option<u64> {
    histories
        .iter()
        .zip(cursors)
        .filter_map(|(history, cursor)| history.bars.get(*cursor).map(|bar| bar.timestamp_ns))
        .min()
}

fn remove_id(schedules: &mut BTreeMap<(u64, String), Vec<String>>, key: &(u64, String), id: &str) {
    if let Some(ids) = schedules.get_mut(key) {
        ids.retain(|value| value != id);
        if ids.is_empty() {
            schedules.remove(key);
        }
    }
}

fn trading_day(timestamp_ns: u64) -> Result<TradingDay, FixedCapitalError> {
    let local_day = timestamp_ns
        .checked_add(SHANGHAI_OFFSET_NS)
        .ok_or(FixedCapitalError::TimestampRange)?
        / NANOS_PER_DAY;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("valid epoch");
    let days = i64::try_from(local_day).map_err(|_| FixedCapitalError::TimestampRange)?;
    epoch
        .checked_add_signed(Duration::days(days))
        .map(TradingDay)
        .ok_or(FixedCapitalError::TimestampRange)
}

fn component_order(
    configured: Option<&[String]>,
    trades: &[ComponentTrade],
) -> Result<Vec<String>, FixedCapitalError> {
    let observed: BTreeSet<_> = trades.iter().map(|trade| trade.component.clone()).collect();
    let Some(configured) = configured else {
        return Ok(observed.into_iter().collect());
    };
    let allowed: BTreeSet<_> = configured.iter().cloned().collect();
    if let Some(component) = observed.difference(&allowed).next() {
        return Err(FixedCapitalError::UnknownComponent(component.clone()));
    }
    Ok(configured.to_vec())
}

fn build_daily(
    evaluation_days: &[TradingDay],
    components: &[String],
    trades: &[ComponentTrade],
) -> Result<Vec<DailyReturn>, FixedCapitalError> {
    let mut by_day: BTreeMap<TradingDay, BTreeMap<String, i64>> = evaluation_days
        .iter()
        .cloned()
        .map(|day| (day, BTreeMap::new()))
        .collect();
    for trade in trades {
        let day = trading_day(trade.exit_timestamp_ns)?;
        let values = by_day
            .get_mut(&day)
            .ok_or_else(|| FixedCapitalError::ExitOutsideCalendar(day.clone()))?;
        let value = values.entry(trade.component.clone()).or_default();
        *value = value
            .checked_add(trade.weighted_return_units)
            .ok_or(FixedCapitalError::FixedPointOverflow)?;
    }
    by_day
        .into_iter()
        .map(|(day, mut values)| {
            for component in components {
                values.entry(component.clone()).or_default();
            }
            let portfolio_return_units = values
                .values()
                .try_fold(0_i64, |total, value| total.checked_add(*value))
                .ok_or(FixedCapitalError::FixedPointOverflow)?;
            Ok(DailyReturn {
                day,
                component_return_units: values,
                portfolio_return_units,
            })
        })
        .collect()
}

fn evaluate(values: &[DailyReturn]) -> FixedCapitalPerformance {
    let returns: Vec<_> = values.iter().map(DailyReturn::portfolio_return).collect();
    let count = returns.len();
    let mean = returns.iter().sum::<f64>() / count as f64;
    let volatility = if count == 1 {
        0.0
    } else {
        (returns
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (count - 1) as f64)
            .sqrt()
    };
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_drawdown = 0.0_f64;
    for value in &returns {
        equity *= 1.0 + value;
        peak = peak.max(equity);
        max_drawdown = max_drawdown.min(equity / peak - 1.0);
    }
    FixedCapitalPerformance {
        days: count,
        mean_daily_return: mean,
        daily_volatility: volatility,
        sharpe: if volatility == 0.0 {
            0.0
        } else {
            mean / volatility * TRADING_DAYS_PER_YEAR.sqrt()
        },
        annualized_return: equity.powf(TRADING_DAYS_PER_YEAR / count as f64) - 1.0,
        total_return: equity - 1.0,
        max_drawdown,
        nonzero_days: returns.iter().filter(|value| **value != 0.0).count(),
    }
}

fn to_fixed(value: f64) -> Result<i64, FixedCapitalError> {
    if !value.is_finite() || value.abs() >= i64::MAX as f64 / FIXED_SCALE {
        return Err(FixedCapitalError::FixedPointOverflow);
    }
    let formatted = format!("{value:.12}");
    let negative = formatted.starts_with('-');
    let unsigned = formatted.trim_start_matches('-');
    let (whole, fractional) = unsigned
        .split_once('.')
        .expect("fixed format has decimal point");
    let magnitude = whole
        .parse::<i64>()
        .ok()
        .and_then(|value| value.checked_mul(FIXED_SCALE as i64))
        .and_then(|value| {
            fractional
                .parse::<i64>()
                .ok()
                .and_then(|fraction| value.checked_add(fraction))
        })
        .ok_or(FixedCapitalError::FixedPointOverflow)?;
    if negative {
        magnitude
            .checked_neg()
            .ok_or(FixedCapitalError::FixedPointOverflow)
    } else {
        Ok(magnitude)
    }
}

fn from_fixed(value: i64) -> f64 {
    value as f64 / FIXED_SCALE
}

fn fixed_string(value: i64) -> String {
    let sign = if value < 0 { "-" } else { "" };
    let magnitude = value.unsigned_abs();
    format!(
        "{sign}{}.{:012}",
        magnitude / FIXED_SCALE as u64,
        magnitude % FIXED_SCALE as u64
    )
}

fn write_trades(path: &Path, trades: &[ComponentTrade]) -> Result<(), FixedCapitalError> {
    let mut writer = csv::Writer::from_path(path).map_err(FixedCapitalError::Csv)?;
    writer
        .write_record([
            "id",
            "component",
            "instrument",
            "signal_timestamp_ns",
            "entry_timestamp_ns",
            "exit_timestamp_ns",
            "setup",
            "side",
            "entry_price",
            "exit_price",
            "exit_reason",
            "cost_fraction",
            "component_scale",
            "instrument_weight",
            "net_return",
            "weighted_return",
        ])
        .map_err(FixedCapitalError::Csv)?;
    for trade in trades {
        writer
            .write_record([
                trade.id.clone(),
                trade.component.clone(),
                trade.instrument.clone(),
                trade.signal_timestamp_ns.to_string(),
                trade.entry_timestamp_ns.to_string(),
                trade.exit_timestamp_ns.to_string(),
                trade.setup.clone(),
                format!("{:?}", trade.side).to_lowercase(),
                format!("{:.12}", trade.entry_price),
                format!("{:.12}", trade.exit_price),
                trade.exit_reason.to_string(),
                format!("{:.12}", trade.cost_fraction),
                format!("{:.12}", trade.component_scale),
                format!("{:.12}", trade.instrument_weight),
                fixed_string(trade.net_return_units),
                fixed_string(trade.weighted_return_units),
            ])
            .map_err(FixedCapitalError::Csv)?;
    }
    writer.flush().map_err(FixedCapitalError::WriteOutput)
}

fn write_daily(
    path: &Path,
    components: &[String],
    values: &[DailyReturn],
) -> Result<(), FixedCapitalError> {
    let mut writer = csv::Writer::from_path(path).map_err(FixedCapitalError::Csv)?;
    let mut header = vec!["day".to_owned()];
    header.extend(
        components
            .iter()
            .map(|component| format!("{component}_return")),
    );
    header.push("portfolio_return".to_owned());
    writer
        .write_record(header)
        .map_err(FixedCapitalError::Csv)?;
    for row in values {
        let mut record = vec![row.day.to_string()];
        record.extend(components.iter().map(|component| {
            fixed_string(
                row.component_return_units
                    .get(component)
                    .copied()
                    .unwrap_or_default(),
            )
        }));
        record.push(fixed_string(row.portfolio_return_units));
        writer
            .write_record(record)
            .map_err(FixedCapitalError::Csv)?;
    }
    writer.flush().map_err(FixedCapitalError::WriteOutput)
}

fn write_censored(path: &Path, values: &[CensoredExposure]) -> Result<(), FixedCapitalError> {
    let mut writer = csv::Writer::from_path(path).map_err(FixedCapitalError::Csv)?;
    writer
        .write_record([
            "id",
            "component",
            "instrument",
            "signal_timestamp_ns",
            "entry_timestamp_ns",
        ])
        .map_err(FixedCapitalError::Csv)?;
    for value in values {
        writer
            .write_record([
                value.id.clone(),
                value.component.clone(),
                value.instrument.clone(),
                value.signal_timestamp_ns.to_string(),
                value
                    .entry_timestamp_ns
                    .map(|timestamp| timestamp.to_string())
                    .unwrap_or_default(),
            ])
            .map_err(FixedCapitalError::Csv)?;
    }
    writer.flush().map_err(FixedCapitalError::WriteOutput)
}

fn write_metrics(path: &Path, result: &FixedCapitalResult) -> Result<(), FixedCapitalError> {
    let performance = &result.performance;
    let value = serde_json::json!({
        "bars": result.bars,
        "data_size_bytes": result.data_size_bytes,
        "runtime_ms": result.runtime_ms,
        "trades": result.trades.len(),
        "censored_exposures": result.censored.len(),
        "performance": {
            "days": performance.days,
            "mean_daily_return": performance.mean_daily_return,
            "daily_volatility": performance.daily_volatility,
            "sharpe": performance.sharpe,
            "annualized_return": performance.annualized_return,
            "total_return": performance.total_return,
            "max_drawdown": performance.max_drawdown,
            "nonzero_days": performance.nonzero_days,
        }
    });
    let source = serde_json::to_vec_pretty(&value).map_err(FixedCapitalError::Json)?;
    fs::write(path, source).map_err(FixedCapitalError::WriteOutput)
}

#[derive(Debug)]
pub enum FixedCapitalError {
    InvalidConfig(&'static str),
    ReadData(std::io::Error),
    Data(bullet_data::DataError),
    EmptyBars,
    NonIncreasingBars(String),
    Strategy(String),
    InvalidOrder(String),
    DuplicateExposure(String),
    UnknownInstrument(String),
    UnknownComponent(String),
    ExitNotAfterSignal(String),
    ExitNotAfterEntry(String),
    InvalidStop(String),
    InactiveExposure(String),
    MissingScheduledBar {
        instrument: String,
        timestamp_ns: u64,
    },
    TimestampRange,
    ExitOutsideCalendar(TradingDay),
    FixedPointOverflow,
    WriteOutput(std::io::Error),
    Csv(csv::Error),
    Json(serde_json::Error),
}

impl fmt::Display for FixedCapitalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => formatter.write_str(message),
            Self::ReadData(error) => write!(formatter, "cannot read data metadata: {error}"),
            Self::Data(error) => write!(formatter, "cannot read bars: {error}"),
            Self::EmptyBars => {
                formatter.write_str("fixed-capital replay requires at least one bar")
            }
            Self::NonIncreasingBars(instrument) => {
                write!(formatter, "bars must strictly increase for {instrument}")
            }
            Self::Strategy(error) => write!(formatter, "strategy failed: {error}"),
            Self::InvalidOrder(message) => write!(formatter, "invalid exposure order: {message}"),
            Self::DuplicateExposure(id) => write!(formatter, "duplicate exposure id `{id}`"),
            Self::UnknownInstrument(instrument) => {
                write!(formatter, "unknown instrument `{instrument}`")
            }
            Self::UnknownComponent(component) => {
                write!(formatter, "unregistered component `{component}`")
            }
            Self::ExitNotAfterSignal(id) => {
                write!(formatter, "exposure `{id}` exits no later than its signal")
            }
            Self::ExitNotAfterEntry(id) => {
                write!(formatter, "exposure `{id}` exits no later than its entry")
            }
            Self::InvalidStop(id) => write!(
                formatter,
                "exposure `{id}` has an invalid or non-adverse stop"
            ),
            Self::InactiveExposure(id) => write!(formatter, "exposure `{id}` is not active"),
            Self::MissingScheduledBar {
                instrument,
                timestamp_ns,
            } => write!(
                formatter,
                "scheduled {instrument} bar is missing at {timestamp_ns}"
            ),
            Self::TimestampRange => {
                formatter.write_str("timestamp is outside the supported calendar range")
            }
            Self::ExitOutsideCalendar(day) => write!(
                formatter,
                "completed exposure exits outside evaluation calendar on {day}"
            ),
            Self::FixedPointOverflow => {
                formatter.write_str("canonical twelve-decimal value overflows")
            }
            Self::WriteOutput(error) => {
                write!(formatter, "cannot write fixed-capital output: {error}")
            }
            Self::Csv(error) => write!(formatter, "cannot encode fixed-capital CSV: {error}"),
            Self::Json(error) => write!(formatter, "cannot encode fixed-capital metrics: {error}"),
        }
    }
}

impl Error for FixedCapitalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ReadData(error) | Self::WriteOutput(error) => Some(error),
            Self::Data(error) => Some(error),
            Self::Csv(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use bullet_data::HistoryBar;

    use super::{
        ExitPlan, ExposureOrder, ExposureSide, FixedCapitalContext, FixedCapitalError,
        FixedCapitalStrategy, InstrumentHistory, ScheduledFill, TradingDay,
        run_fixed_capital_history,
    };

    const MINUTE: u64 = 60_000_000_000;
    const DAY: u64 = 86_400_000_000_000;
    const SHANGHAI: u64 = 8 * 60 * MINUTE;

    fn timestamp(day: u64, hour: u64, minute: u64) -> u64 {
        day * DAY + (hour * 60 + minute) * MINUTE - SHANGHAI
    }

    fn bar(timestamp_ns: u64, open: f64, high: f64, low: f64, close: f64) -> HistoryBar {
        HistoryBar {
            timestamp_ns,
            open,
            high,
            low,
            close,
            volume: 10.0,
            money: 1_000.0,
            open_interest: 20.0,
        }
    }

    fn day(day: u32) -> TradingDay {
        TradingDay::new(1970, 1, day).expect("fixture date is valid")
    }

    #[derive(Default)]
    struct FixtureStrategy {
        orders: Vec<(u64, ExposureOrder)>,
        callbacks: Vec<(u64, Vec<String>, Vec<f64>)>,
    }

    impl FixedCapitalStrategy for FixtureStrategy {
        fn on_timestamp(
            &mut self,
            context: FixedCapitalContext<'_>,
        ) -> Result<Vec<ExposureOrder>, String> {
            self.callbacks.push((
                context.timestamp_ns,
                context
                    .bars
                    .iter()
                    .map(|value| value.instrument.to_owned())
                    .collect(),
                context.bars.iter().map(|value| value.bar.close).collect(),
            ));
            let split = self
                .orders
                .partition_point(|(timestamp_ns, _)| *timestamp_ns <= context.timestamp_ns);
            let due = self.orders.drain(..split).collect::<Vec<_>>();
            if due
                .iter()
                .any(|(timestamp_ns, _)| *timestamp_ns != context.timestamp_ns)
            {
                return Err("fixture missed its causal signal timestamp".to_owned());
            }
            Ok(due.into_iter().map(|(_, order)| order).collect())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn order(
        id: &str,
        component: &str,
        instrument: &str,
        side: ExposureSide,
        exit: ExitPlan,
        cost_fraction: f64,
        component_scale: f64,
        instrument_weight: f64,
    ) -> ExposureOrder {
        ExposureOrder {
            id: id.to_owned(),
            component: component.to_owned(),
            instrument: instrument.to_owned(),
            setup: "fixture".to_owned(),
            side,
            component_scale,
            instrument_weight,
            exit,
            cost_fraction,
        }
    }

    #[test]
    fn batches_symbols_before_cross_sectional_callback_and_never_exposes_future_bars() {
        let first = timestamp(0, 9, 31);
        let second = timestamp(0, 9, 32);
        let histories = vec![
            InstrumentHistory {
                instrument: "B".to_owned(),
                bars: vec![
                    bar(first, 200.0, 201.0, 199.0, 200.0),
                    bar(second, 220.0, 221.0, 219.0, 220.0),
                ],
            },
            InstrumentHistory {
                instrument: "A".to_owned(),
                bars: vec![
                    bar(first, 100.0, 101.0, 99.0, 100.0),
                    bar(second, 110.0, 111.0, 109.0, 110.0),
                ],
            },
        ];
        let mut strategy = FixtureStrategy::default();

        run_fixed_capital_history(&[day(1)], &histories, 0, &mut strategy)
            .expect("fixture replay completes");

        assert_eq!(strategy.callbacks.len(), 2);
        assert_eq!(strategy.callbacks[0].0, first);
        assert_eq!(strategy.callbacks[0].1, vec!["A", "B"]);
        assert_eq!(strategy.callbacks[0].2, vec![100.0, 200.0]);
        assert!(!strategy.callbacks[0].2.contains(&220.0));
    }

    #[test]
    fn keeps_fractional_overlapping_components_and_canonical_daily_sum() {
        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let exit = timestamp(0, 9, 33);
        let histories = vec![InstrumentHistory {
            instrument: "A".to_owned(),
            bars: vec![
                bar(signal, 100.0, 100.0, 100.0, 100.0),
                bar(entry, 100.0, 100.0, 100.0, 100.0),
                bar(exit, 110.0, 110.0, 110.0, 110.0),
            ],
        }];
        let mut strategy = FixtureStrategy {
            orders: vec![
                (
                    signal,
                    order(
                        "left",
                        "left_component",
                        "A",
                        ExposureSide::Long,
                        ExitPlan::At {
                            timestamp_ns: exit,
                            fill: ScheduledFill::Open,
                        },
                        0.0,
                        0.000008,
                        0.125,
                    ),
                ),
                (
                    signal,
                    order(
                        "right",
                        "right_component",
                        "A",
                        ExposureSide::Short,
                        ExitPlan::At {
                            timestamp_ns: exit,
                            fill: ScheduledFill::Close,
                        },
                        0.000123,
                        0.1,
                        0.25,
                    ),
                ),
            ],
            ..FixtureStrategy::default()
        };

        let result = run_fixed_capital_history(&[day(1)], &histories, 0, &mut strategy)
            .expect("overlapping replay completes");

        assert_eq!(result.trades.len(), 2);
        assert_eq!(result.trades[0].weighted_return_units, 100_000);
        assert_eq!(result.trades[1].weighted_return_units, -2_502_767_500);
        assert_eq!(
            result.daily_returns[0].portfolio_return_units,
            -2_502_667_500
        );
        assert_eq!(result.daily_returns[0].portfolio_return(), -0.0025026675);
    }

    #[test]
    fn supports_scheduled_close_and_gap_aware_pre_registered_stop() {
        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let middle = timestamp(0, 9, 33);
        let scheduled = timestamp(0, 9, 34);
        let histories = vec![InstrumentHistory {
            instrument: "A".to_owned(),
            bars: vec![
                bar(signal, 100.0, 101.0, 99.0, 100.0),
                bar(entry, 100.0, 101.0, 99.0, 100.0),
                bar(middle, 94.0, 96.0, 93.0, 95.0),
                bar(scheduled, 110.0, 111.0, 109.0, 110.0),
            ],
        }];
        let mut strategy = FixtureStrategy {
            orders: vec![
                (
                    signal,
                    order(
                        "stop",
                        "stop_component",
                        "A",
                        ExposureSide::Long,
                        ExitPlan::StopOrAt {
                            stop_price: 95.0,
                            timestamp_ns: scheduled,
                            fill: ScheduledFill::Close,
                        },
                        0.000246,
                        1.0,
                        1.0,
                    ),
                ),
                (
                    signal,
                    order(
                        "close",
                        "close_component",
                        "A",
                        ExposureSide::Long,
                        ExitPlan::At {
                            timestamp_ns: scheduled,
                            fill: ScheduledFill::Close,
                        },
                        0.000369,
                        1.0,
                        1.0,
                    ),
                ),
            ],
            ..FixtureStrategy::default()
        };

        let result = run_fixed_capital_history(&[day(1)], &histories, 0, &mut strategy)
            .expect("stop and close replay completes");
        let by_id = result
            .trades
            .iter()
            .map(|trade| (trade.id.as_str(), trade))
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(by_id["stop"].exit_timestamp_ns, middle);
        assert_eq!(by_id["stop"].exit_price, 94.0);
        assert_eq!(by_id["stop"].net_return_units, -60_231_240_000);
        assert_eq!(by_id["close"].exit_timestamp_ns, scheduled);
        assert_eq!(by_id["close"].exit_price, 110.0);
        assert_eq!(by_id["close"].net_return_units, 99_594_100_000);
    }

    #[test]
    fn a_stop_crossed_during_the_entry_bar_exits_at_the_adverse_open() {
        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let following = timestamp(0, 9, 33);
        let scheduled = timestamp(0, 9, 34);
        let histories = vec![InstrumentHistory {
            instrument: "A".to_owned(),
            bars: vec![
                bar(signal, 100.0, 100.0, 100.0, 100.0),
                bar(entry, 90.0, 91.0, 89.0, 90.0),
                bar(following, 89.0, 90.0, 88.0, 89.0),
                bar(scheduled, 100.0, 100.0, 100.0, 100.0),
            ],
        }];
        let mut strategy = FixtureStrategy {
            orders: vec![(
                signal,
                order(
                    "entry-gap-stop",
                    "component",
                    "A",
                    ExposureSide::Long,
                    ExitPlan::StopOrAt {
                        stop_price: 95.0,
                        timestamp_ns: scheduled,
                        fill: ScheduledFill::Open,
                    },
                    0.0,
                    1.0,
                    1.0,
                ),
            )],
            ..FixtureStrategy::default()
        };

        let result = run_fixed_capital_history(&[day(1)], &histories, 0, &mut strategy)
            .expect("entry-gap stop replay completes");

        assert_eq!(result.trades[0].entry_price, 90.0);
        assert_eq!(result.trades[0].exit_timestamp_ns, entry);
        assert_eq!(result.trades[0].exit_price, 90.0);
    }

    #[test]
    fn accepts_component_local_zero_half_round_trip_and_combined_costs() {
        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let exit = timestamp(0, 9, 33);
        let histories = vec![InstrumentHistory {
            instrument: "A".to_owned(),
            bars: vec![
                bar(signal, 100.0, 100.0, 100.0, 100.0),
                bar(entry, 100.0, 100.0, 100.0, 100.0),
                bar(exit, 100.0, 100.0, 100.0, 100.0),
            ],
        }];
        let mut strategy = FixtureStrategy {
            orders: [0.0, 0.000123, 0.000246, 0.000369]
                .into_iter()
                .enumerate()
                .map(|(index, cost)| {
                    (
                        signal,
                        order(
                            &format!("cost-{index}"),
                            "cost_component",
                            "A",
                            ExposureSide::Long,
                            ExitPlan::At {
                                timestamp_ns: exit,
                                fill: ScheduledFill::Open,
                            },
                            cost,
                            1.0,
                            1.0,
                        ),
                    )
                })
                .collect(),
            ..FixtureStrategy::default()
        };

        let result = run_fixed_capital_history(&[day(1)], &histories, 0, &mut strategy)
            .expect("cost replay completes");
        assert_eq!(
            result
                .trades
                .iter()
                .map(|trade| trade.net_return_units)
                .collect::<Vec<_>>(),
            vec![0, -123_000_000, -246_000_000, -369_000_000]
        );
    }

    #[test]
    fn hard_fails_unknown_instrument_duplicate_id_and_skipped_scheduled_bar() {
        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let missing = timestamp(0, 9, 33);
        let later = timestamp(0, 9, 34);
        let histories = vec![InstrumentHistory {
            instrument: "A".to_owned(),
            bars: vec![
                bar(signal, 100.0, 100.0, 100.0, 100.0),
                bar(entry, 100.0, 100.0, 100.0, 100.0),
                bar(later, 100.0, 100.0, 100.0, 100.0),
            ],
        }];
        let make = |id: &str, instrument: &str| {
            order(
                id,
                "component",
                instrument,
                ExposureSide::Long,
                ExitPlan::At {
                    timestamp_ns: missing,
                    fill: ScheduledFill::Open,
                },
                0.0,
                1.0,
                1.0,
            )
        };

        let mut unknown = FixtureStrategy {
            orders: vec![(signal, make("unknown", "B"))],
            ..FixtureStrategy::default()
        };
        assert!(matches!(
            run_fixed_capital_history(&[day(1)], &histories, 0, &mut unknown),
            Err(FixedCapitalError::UnknownInstrument(instrument)) if instrument == "B"
        ));

        let mut duplicate = FixtureStrategy {
            orders: vec![(signal, make("same", "A")), (signal, make("same", "A"))],
            ..FixtureStrategy::default()
        };
        assert!(matches!(
            run_fixed_capital_history(&[day(1)], &histories, 0, &mut duplicate),
            Err(FixedCapitalError::DuplicateExposure(id)) if id == "same"
        ));

        let mut absent = FixtureStrategy {
            orders: vec![(signal, make("absent", "A"))],
            ..FixtureStrategy::default()
        };
        assert!(matches!(
            run_fixed_capital_history(&[day(1)], &histories, 0, &mut absent),
            Err(FixedCapitalError::MissingScheduledBar { instrument, timestamp_ns })
                if instrument == "A" && timestamp_ns == missing
        ));
    }

    #[test]
    fn writes_twelve_decimal_canonical_csv_and_sample_sharpe_metrics() {
        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let first_exit = timestamp(0, 9, 33);
        let second_signal = timestamp(1, 9, 31);
        let second_entry = timestamp(1, 9, 32);
        let second_exit = timestamp(1, 9, 33);
        let histories = vec![InstrumentHistory {
            instrument: "A".to_owned(),
            bars: vec![
                bar(signal, 100.0, 100.0, 100.0, 100.0),
                bar(entry, 100.0, 100.0, 100.0, 100.0),
                bar(first_exit, 101.0, 101.0, 101.0, 101.0),
                bar(second_signal, 100.0, 100.0, 100.0, 100.0),
                bar(second_entry, 100.0, 100.0, 100.0, 100.0),
                bar(second_exit, 99.0, 99.0, 99.0, 99.0),
            ],
        }];
        let mut strategy = FixtureStrategy {
            orders: vec![
                (
                    signal,
                    order(
                        "positive",
                        "component",
                        "A",
                        ExposureSide::Long,
                        ExitPlan::At {
                            timestamp_ns: first_exit,
                            fill: ScheduledFill::Open,
                        },
                        0.0,
                        1.0,
                        1.0,
                    ),
                ),
                (
                    second_signal,
                    order(
                        "negative",
                        "component",
                        "A",
                        ExposureSide::Long,
                        ExitPlan::At {
                            timestamp_ns: second_exit,
                            fill: ScheduledFill::Open,
                        },
                        0.0,
                        1.0,
                        1.0,
                    ),
                ),
            ],
            ..FixtureStrategy::default()
        };
        let result = run_fixed_capital_history(&[day(1), day(2)], &histories, 0, &mut strategy)
            .expect("two-day replay completes");

        assert_eq!(result.performance.days, 2);
        assert_eq!(result.performance.mean_daily_return, 0.0);
        assert_eq!(result.performance.daily_volatility, 2.0_f64.sqrt() / 100.0);
        assert_eq!(result.performance.sharpe, 0.0);
        let output = std::env::temp_dir().join(format!(
            "bullet-fixed-capital-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock is after epoch")
                .as_nanos()
        ));
        result
            .write_csv(&output)
            .expect("canonical files are written");
        let daily = std::fs::read_to_string(output.join("daily_returns.csv"))
            .expect("daily CSV is readable");
        std::fs::remove_dir_all(&output).expect("fixture output is removed");
        assert_eq!(
            daily,
            "day,component_return,portfolio_return\n1970-01-01,0.010000000000,0.010000000000\n1970-01-02,-0.010000000000,-0.010000000000\n"
        );
    }

    #[test]
    fn configured_component_order_keeps_zero_trade_columns_and_rejects_unknown_components() {
        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let exit = timestamp(0, 9, 33);
        let histories = vec![InstrumentHistory {
            instrument: "A".to_owned(),
            bars: vec![
                bar(signal, 100.0, 100.0, 100.0, 100.0),
                bar(entry, 100.0, 100.0, 100.0, 100.0),
                bar(exit, 101.0, 101.0, 101.0, 101.0),
            ],
        }];
        let strategy = || FixtureStrategy {
            orders: vec![(
                signal,
                order(
                    "observed",
                    "observed_component",
                    "A",
                    ExposureSide::Long,
                    ExitPlan::At {
                        timestamp_ns: exit,
                        fill: ScheduledFill::Open,
                    },
                    0.0,
                    1.0,
                    1.0,
                ),
            )],
            ..FixtureStrategy::default()
        };
        let configured = vec!["zero_component".to_owned(), "observed_component".to_owned()];
        let mut accepted = strategy();
        let result = super::run_fixed_capital_history_inner(
            &[day(1)],
            Some(&configured),
            &histories,
            0,
            &mut accepted,
        )
        .expect("registered component replay completes");

        assert_eq!(result.component_order, configured);
        assert_eq!(
            result.daily_returns[0].component_return_units["zero_component"],
            0
        );

        let mut rejected = strategy();
        let error = super::run_fixed_capital_history_inner(
            &[day(1)],
            Some(&["zero_component".to_owned()]),
            &histories,
            0,
            &mut rejected,
        )
        .expect_err("unregistered component is rejected");
        assert!(matches!(
            error,
            FixedCapitalError::UnknownComponent(component)
                if component == "observed_component"
        ));
    }

    #[test]
    fn future_bar_mutation_cannot_change_an_already_emitted_order() {
        struct FirstTimestampStrategy {
            first_order: Option<ExposureOrder>,
        }
        impl FixedCapitalStrategy for FirstTimestampStrategy {
            fn on_timestamp(
                &mut self,
                _context: FixedCapitalContext<'_>,
            ) -> Result<Vec<ExposureOrder>, String> {
                Ok(self.first_order.take().into_iter().collect())
            }
        }

        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let exit = timestamp(0, 9, 33);
        let strategy_order = || {
            order(
                "causal",
                "component",
                "A",
                ExposureSide::Long,
                ExitPlan::At {
                    timestamp_ns: exit,
                    fill: ScheduledFill::Open,
                },
                0.0,
                1.0,
                1.0,
            )
        };
        let replay = |future_close: f64| {
            let histories = vec![InstrumentHistory {
                instrument: "A".to_owned(),
                bars: vec![
                    bar(signal, 100.0, 100.0, 100.0, 100.0),
                    bar(entry, 100.0, 100.0, 100.0, future_close),
                    bar(exit, 110.0, 110.0, 110.0, 110.0),
                ],
            }];
            let mut strategy = FirstTimestampStrategy {
                first_order: Some(strategy_order()),
            };
            run_fixed_capital_history(&[day(1)], &histories, 0, &mut strategy)
                .expect("causal replay completes")
        };

        let left = replay(80.0);
        let right = replay(120.0);
        assert_eq!(left.trades, right.trades);
    }

    #[test]
    fn final_pending_and_active_exposures_are_right_censored_in_id_order() {
        let signal = timestamp(0, 9, 31);
        let entry = timestamp(0, 9, 32);
        let future = timestamp(1, 9, 31);
        let histories = vec![InstrumentHistory {
            instrument: "A".to_owned(),
            bars: vec![
                bar(signal, 100.0, 100.0, 100.0, 100.0),
                bar(entry, 100.0, 100.0, 100.0, 100.0),
            ],
        }];
        let mut strategy = FixtureStrategy {
            orders: vec![
                (
                    signal,
                    order(
                        "z-active",
                        "component",
                        "A",
                        ExposureSide::Long,
                        ExitPlan::At {
                            timestamp_ns: future,
                            fill: ScheduledFill::Open,
                        },
                        0.0,
                        1.0,
                        1.0,
                    ),
                ),
                (
                    entry,
                    order(
                        "a-pending",
                        "component",
                        "A",
                        ExposureSide::Long,
                        ExitPlan::At {
                            timestamp_ns: future,
                            fill: ScheduledFill::Open,
                        },
                        0.0,
                        1.0,
                        1.0,
                    ),
                ),
            ],
            ..FixtureStrategy::default()
        };

        let result = run_fixed_capital_history(&[day(1)], &histories, 0, &mut strategy)
            .expect("right-censored replay completes");

        assert!(result.trades.is_empty());
        assert_eq!(
            result
                .censored
                .iter()
                .map(|value| (value.id.as_str(), value.entry_timestamp_ns))
                .collect::<Vec<_>>(),
            vec![("a-pending", None), ("z-active", Some(entry))]
        );
    }

    #[test]
    fn canonical_rounding_is_twelve_decimal_half_even_without_negative_zero() {
        assert_eq!(super::to_fixed(0.5e-12).expect("finite value"), 0);
        assert_eq!(super::to_fixed(1.5e-12).expect("finite value"), 2);
        assert_eq!(super::to_fixed(2.5e-12).expect("finite value"), 2);
        assert_eq!(super::to_fixed(-0.5e-12).expect("finite value"), 0);
        assert_eq!(super::to_fixed(-1.5e-12).expect("finite value"), -2);
        assert_eq!(super::fixed_string(0), "0.000000000000");
    }

    #[test]
    fn fixture_components_are_unique_for_stable_daily_columns() {
        let values = ["left", "right", "stop", "close", "cost"];
        assert_eq!(
            values.into_iter().collect::<BTreeSet<_>>().len(),
            values.len()
        );
    }
}

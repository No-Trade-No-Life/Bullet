use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

use bullet_data::HistoryBar;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use serde::Deserialize;

use crate::config::InstrumentConfig;

const MIN_COMPLETED_ROWS: usize = 60;
const EXIT_OFFSET_FROM_SIGNAL: usize = 20;
const NANOSECONDS_PER_MINUTE: u64 = 60_000_000_000;

/// CTPD's `event: tick` JSON payload. `action_day`, rather than CTP's
/// trading-day accounting label, is the calendar-day boundary used by the
/// lab-0344 Parquet `date.normalize()` rule.
#[derive(Clone, Debug, Deserialize)]
pub struct CtpdTick {
    pub instrument_id: String,
    pub exchange_id: String,
    pub trading_day: String,
    pub action_day: String,
    pub update_time: String,
    pub update_millisec: i32,
    pub last_price: f64,
    pub open_interest: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Side {
    Long,
    Short,
}

/// Parquet `date` labels the completed one-minute bar's endpoint. The live
/// assembler assigns the same endpoint to a bar opened by a tick at its start.
#[derive(Clone, Debug)]
struct CompletedBar {
    timestamp_ns: u64,
    close: f64,
    open_interest: f64,
}

#[derive(Clone, Debug)]
struct OpenPosition {
    side: Side,
    entry_price: f64,
    exit_row: usize,
}

#[derive(Clone, Debug)]
struct PendingEntry {
    side: Side,
    entry_row: usize,
    exit_row: usize,
}

/// The one visible target position for one continuous-series strategy leg.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetPosition {
    pub symbol: String,
    pub ctpd_instrument_id: String,
    pub exchange_id: String,
    pub contracts: i64,
    pub entry_price: f64,
    pub latest_price: f64,
    pub multiplier: f64,
    pub updated_at_ms: i64,
}

#[derive(Clone, Debug)]
pub struct Lab0344Model {
    symbol: String,
    ctpd_instrument_id: String,
    target_contracts: i64,
    multiplier: f64,
    session_bar_count: usize,
    last_executable_signal_time: NaiveTime,
    action_day: Option<String>,
    completed_rows: usize,
    signal_seen: bool,
    previous: Option<CompletedBar>,
    current: Option<CompletedBar>,
    pending_entry: Option<PendingEntry>,
    position: Option<OpenPosition>,
    exchange_id: String,
    latest_price: Option<f64>,
    last_live_tick_ns: Option<u64>,
    seeded_through_ns: Option<u64>,
}

impl Lab0344Model {
    pub fn new(config: &InstrumentConfig) -> Self {
        Self {
            symbol: config.symbol.clone(),
            ctpd_instrument_id: config.ctpd_instrument_id.clone(),
            target_contracts: config.target_contracts,
            multiplier: config.contract_multiplier,
            session_bar_count: config.session_bar_count,
            last_executable_signal_time: NaiveTime::parse_from_str(
                &config.last_executable_signal_time,
                "%H:%M:%S",
            )
            .expect("InstrumentConfig must contain an HH:MM:SS signal cutoff"),
            action_day: None,
            completed_rows: 0,
            signal_seen: false,
            previous: None,
            current: None,
            pending_entry: None,
            position: None,
            exchange_id: String::new(),
            latest_price: None,
            last_live_tick_ns: None,
            seeded_through_ns: None,
        }
    }

    /// Replays every completed bar in the final natural day from Parquet. This
    /// restores the same first-signal, pending-entry, and open-position state
    /// that the causal live state machine would have produced before startup.
    pub fn seed_history(&mut self, history: &[HistoryBar]) -> Result<(), String> {
        self.clear_state();
        let Some(last) = history.last() else {
            return Ok(());
        };
        let day = history_day(last.timestamp_ns)?;
        let mut session = Vec::new();
        for bar in history {
            if history_day(bar.timestamp_ns)? == day {
                session.push(bar);
            }
        }
        if session.len() > self.session_bar_count {
            return Err(format!(
                "Parquet session has {} bars, above configured session_bar_count {}",
                session.len(),
                self.session_bar_count
            ));
        }

        self.reset_day(day);
        for bar in session {
            self.open_next_bar(bar.open);
            self.complete_bar(CompletedBar {
                timestamp_ns: bar.timestamp_ns,
                close: bar.close,
                open_interest: bar.open_interest,
            })?;
        }
        self.seeded_through_ns = Some(last.timestamp_ns);
        // A Parquet-only state is not a fresh CTPD target. The first advancing
        // live tick both proves feed liveness and publishes any carried target.
        self.exchange_id.clear();
        self.latest_price = None;
        Ok(())
    }

    pub fn ingest(&mut self, tick: CtpdTick) -> Result<Option<TargetPosition>, String> {
        validate_tick(&tick, &self.ctpd_instrument_id)?;
        let tick_timestamp_ns = tick_timestamp_ns(&tick)?;
        let bar_endpoint_ns = bar_endpoint_ns(tick_timestamp_ns)?;

        if self.action_day.as_deref() != Some(tick.action_day.as_str()) {
            // A calendar-day change is an intraday hard boundary. Do not use a
            // next-day tick to complete a previous day's unfinished bar.
            self.reset_day(tick.action_day.clone());
        }
        if self
            .seeded_through_ns
            .is_some_and(|seeded| bar_endpoint_ns <= seeded)
        {
            // CTPD can initially replay the already-persisted final bar. It
            // cannot change state and is ignored until a bar advances it.
            return Ok(None);
        }
        if self
            .last_live_tick_ns
            .is_some_and(|previous| tick_timestamp_ns < previous)
        {
            return Err("CTPD tick is out of order".into());
        }

        match self.current.as_mut() {
            None => {
                self.open_next_bar(tick.last_price);
                self.current = Some(CompletedBar {
                    timestamp_ns: bar_endpoint_ns,
                    close: tick.last_price,
                    open_interest: tick.open_interest,
                });
            }
            Some(current) if current.timestamp_ns == bar_endpoint_ns => {
                current.close = tick.last_price;
                current.open_interest = tick.open_interest;
            }
            Some(current) if current.timestamp_ns < bar_endpoint_ns => {
                let completed = self.current.take().expect("current bar exists");
                self.complete_bar(completed)?;
                self.open_next_bar(tick.last_price);
                self.current = Some(CompletedBar {
                    timestamp_ns: bar_endpoint_ns,
                    close: tick.last_price,
                    open_interest: tick.open_interest,
                });
            }
            Some(_) => return Err("CTPD tick moves before the current minute".into()),
        }

        self.last_live_tick_ns = Some(tick_timestamp_ns);
        self.exchange_id = tick.exchange_id;
        self.latest_price = Some(tick.last_price);
        self.target()
    }

    /// A feed gap invalidates both the visible target and the accumulated bar
    /// state. Reconnect starts with fresh complete bars instead of joining two
    /// unknown market intervals.
    pub fn clear_state(&mut self) {
        self.action_day = None;
        self.completed_rows = 0;
        self.signal_seen = false;
        self.previous = None;
        self.current = None;
        self.pending_entry = None;
        self.position = None;
        self.exchange_id.clear();
        self.latest_price = None;
        self.last_live_tick_ns = None;
        self.seeded_through_ns = None;
    }

    fn reset_day(&mut self, action_day: String) {
        self.clear_state();
        self.action_day = Some(action_day);
    }

    fn open_next_bar(&mut self, entry_price: f64) {
        let row = self.completed_rows + 1;
        if self
            .position
            .as_ref()
            .is_some_and(|position| position.exit_row == row)
        {
            self.position = None;
        }
        if self
            .pending_entry
            .as_ref()
            .is_some_and(|entry| entry.entry_row == row)
        {
            let entry = self
                .pending_entry
                .take()
                .expect("checked pending entry exists");
            self.position = Some(OpenPosition {
                side: entry.side,
                entry_price,
                exit_row: entry.exit_row,
            });
        }
    }

    fn complete_bar(&mut self, bar: CompletedBar) -> Result<(), String> {
        if self.completed_rows >= self.session_bar_count {
            return Err(format!(
                "received more than configured session_bar_count {}",
                self.session_bar_count
            ));
        }
        self.completed_rows += 1;
        let Some(previous) = self.previous.as_ref() else {
            self.previous = Some(bar);
            return Ok(());
        };
        let can_schedule = self.completed_rows >= MIN_COMPLETED_ROWS
            && self.completed_rows + EXIT_OFFSET_FROM_SIGNAL <= self.session_bar_count
            && bar_time(bar.timestamp_ns)? <= self.last_executable_signal_time;
        if !self.signal_seen && can_schedule {
            let side = if bar.open_interest > previous.open_interest && bar.close > previous.close {
                Some(Side::Long)
            } else if bar.open_interest > previous.open_interest && bar.close < previous.close {
                Some(Side::Short)
            } else {
                None
            };
            if let Some(side) = side {
                self.signal_seen = true;
                self.pending_entry = Some(PendingEntry {
                    side,
                    entry_row: self.completed_rows + 1,
                    exit_row: self.completed_rows + EXIT_OFFSET_FROM_SIGNAL,
                });
            }
        }
        self.previous = Some(bar);
        Ok(())
    }

    fn target(&self) -> Result<Option<TargetPosition>, String> {
        let Some(position) = self.position.as_ref() else {
            return Ok(None);
        };
        let Some(latest_price) = self.latest_price else {
            return Ok(None);
        };
        let contracts = match position.side {
            Side::Long => self.target_contracts,
            Side::Short => -self.target_contracts,
        };
        let exposure = contracts as f64 * latest_price * self.multiplier;
        let floating_profit =
            (latest_price - position.entry_price) * contracts as f64 * self.multiplier;
        if !exposure.is_finite() || !floating_profit.is_finite() {
            return Err("target position arithmetic is not finite".into());
        }
        Ok(Some(TargetPosition {
            symbol: self.symbol.clone(),
            ctpd_instrument_id: self.ctpd_instrument_id.clone(),
            exchange_id: self.exchange_id.clone(),
            contracts,
            entry_price: position.entry_price,
            latest_price,
            multiplier: self.multiplier,
            updated_at_ms: now_ms(),
        }))
    }
}

#[derive(Default)]
pub struct Portfolio {
    models: BTreeMap<String, Lab0344Model>,
    targets: BTreeMap<String, TargetPosition>,
}

impl Portfolio {
    pub fn insert(&mut self, instrument_id: String, model: Lab0344Model) {
        self.models.insert(instrument_id, model);
    }

    pub fn ingest(&mut self, tick: CtpdTick) -> Result<(), String> {
        let id = tick.instrument_id.clone();
        let target = self
            .models
            .get_mut(&id)
            .ok_or_else(|| format!("no configured model for CTPD instrument {id}"))?
            .ingest(tick)?;
        match target {
            Some(target) => {
                self.targets.insert(id, target);
            }
            None => {
                self.targets.remove(&id);
            }
        }
        Ok(())
    }

    /// A disconnected or stale feed must not leave an actionable target
    /// visible to 1Exchange. This is intentionally fail-closed.
    pub fn clear(&mut self, instrument_id: &str) {
        if let Some(model) = self.models.get_mut(instrument_id) {
            model.clear_state();
        }
        self.targets.remove(instrument_id);
    }

    pub fn targets(&self) -> Vec<TargetPosition> {
        self.targets.values().cloned().collect()
    }
}

fn validate_tick(tick: &CtpdTick, expected_instrument: &str) -> Result<(), String> {
    if tick.instrument_id != expected_instrument {
        return Err(format!(
            "CTPD tick instrument {} does not match {expected_instrument}",
            tick.instrument_id
        ));
    }
    if tick.exchange_id.is_empty()
        || NaiveDate::parse_from_str(&tick.trading_day, "%Y%m%d").is_err()
        || !tick.last_price.is_finite()
        || tick.last_price <= 0.0
        || !tick.open_interest.is_finite()
        || tick.open_interest < 0.0
        || !(0..1000).contains(&tick.update_millisec)
    {
        return Err("CTPD tick has invalid market fields".into());
    }
    Ok(())
}

fn tick_timestamp_ns(tick: &CtpdTick) -> Result<u64, String> {
    let datetime = NaiveDateTime::parse_from_str(
        &format!("{} {}", tick.action_day, tick.update_time),
        "%Y%m%d %H:%M:%S",
    )
    .map_err(|_| "CTPD tick has invalid action_day/update_time")?;
    let seconds = datetime.and_utc().timestamp();
    let nanoseconds = i64::from(tick.update_millisec) * 1_000_000;
    let total = seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanoseconds))
        .ok_or("CTPD tick timestamp overflow")?;
    u64::try_from(total).map_err(|_| "CTPD tick timestamp precedes Unix epoch".into())
}

fn bar_endpoint_ns(tick_timestamp_ns: u64) -> Result<u64, String> {
    tick_timestamp_ns
        .checked_div(NANOSECONDS_PER_MINUTE)
        .and_then(|minute| minute.checked_add(1))
        .and_then(|minute| minute.checked_mul(NANOSECONDS_PER_MINUTE))
        .ok_or_else(|| "CTPD bar endpoint overflow".into())
}

/// E-Works reads Parquet `date` as a timezone-naive China-market timestamp.
/// Keep that natural date rather than shifting it through CTP accounting time.
fn history_datetime(timestamp_ns: u64) -> Result<NaiveDateTime, String> {
    let seconds = i64::try_from(timestamp_ns / 1_000_000_000)
        .map_err(|_| "historical timestamp overflows chrono")?;
    let nanoseconds = (timestamp_ns % 1_000_000_000) as u32;
    DateTime::<Utc>::from_timestamp(seconds, nanoseconds)
        .map(|value| value.naive_utc())
        .ok_or_else(|| "historical timestamp is invalid".into())
}

fn history_day(timestamp_ns: u64) -> Result<String, String> {
    Ok(history_datetime(timestamp_ns)?.format("%Y%m%d").to_string())
}

fn bar_time(timestamp_ns: u64) -> Result<NaiveTime, String> {
    Ok(history_datetime(timestamp_ns)?.time())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use bullet_data::HistoryBar;
    use chrono::{Duration, NaiveDateTime};

    use super::{CompletedBar, CtpdTick, Lab0344Model, Portfolio, Side};
    use crate::config::InstrumentConfig;

    fn config() -> InstrumentConfig {
        InstrumentConfig {
            symbol: "IF".into(),
            ctpd_instrument_id: "IF2609".into(),
            parquet: "/unused".into(),
            target_contracts: 2,
            contract_multiplier: 300.0,
            session_bar_count: 240,
            last_executable_signal_time: "14:40:00".into(),
        }
    }

    fn datetime(text: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(text, "%Y%m%d %H:%M:%S").unwrap()
    }

    fn tick_at(at: NaiveDateTime, price: f64, oi: f64) -> CtpdTick {
        CtpdTick {
            instrument_id: "IF2609".into(),
            exchange_id: "CFFEX".into(),
            trading_day: "20260810".into(),
            action_day: at.format("%Y%m%d").to_string(),
            update_time: at.format("%H:%M:%S").to_string(),
            update_millisec: 0,
            last_price: price,
            open_interest: oi,
        }
    }

    fn session_tick(row: usize, price: f64, oi: f64) -> CtpdTick {
        tick_at(
            datetime("20260810 09:30:00") + Duration::minutes(row as i64),
            price,
            oi,
        )
    }

    #[test]
    fn signal_enters_next_bar_and_exits_at_lab0344_offset() {
        let mut model = Lab0344Model::new(&config());
        for row in 0..=60 {
            let oi = if row == 59 { 100.0 } else { 99.0 };
            let target = model
                .ingest(session_tick(row, 4_000.0 + row as f64, oi))
                .unwrap();
            if row < 60 {
                assert!(target.is_none());
            } else {
                assert_eq!(target.unwrap().contracts, 2);
            }
        }
        for row in 61..79 {
            assert!(
                model
                    .ingest(session_tick(row, 4_100.0, 101.0))
                    .unwrap()
                    .is_some()
            );
        }
        // Signal row 60 exits on the opening tick of row 80. No future close
        // is read to perform that exit.
        assert!(
            model
                .ingest(session_tick(79, 4_100.0, 101.0))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn parquet_session_replay_restores_first_signal_and_open_position() {
        let mut model = Lab0344Model::new(&config());
        let first_endpoint = datetime("20260810 09:31:00")
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap() as u64;
        let history = (0..70)
            .map(|row| HistoryBar {
                timestamp_ns: first_endpoint + row * super::NANOSECONDS_PER_MINUTE,
                open: 4_000.0 + row as f64,
                close: 4_000.0 + row as f64,
                open_interest: if row == 59 { 100.0 } else { 99.0 },
            })
            .collect::<Vec<_>>();
        model.seed_history(&history).unwrap();
        assert!(model.signal_seen);
        assert_eq!(model.position.as_ref().unwrap().entry_price, 4_060.0);

        // History ends at 10:40. The live tick at 10:40 begins a 10:41-end
        // bar, advances the historical state, and publishes the carried target.
        let target = model
            .ingest(tick_at(datetime("20260810 10:40:00"), 4_080.0, 101.0))
            .unwrap()
            .unwrap();
        assert_eq!(target.contracts, 2);
        assert_eq!(target.entry_price, 4_060.0);
    }

    #[test]
    fn session_boundary_prevents_a_nonexistent_tail_candidate() {
        let mut model = Lab0344Model::new(&config());
        model.action_day = Some("20260810".into());
        model.completed_rows = 219;
        model.previous = Some(CompletedBar {
            timestamp_ns: datetime("20260810 14:39:00")
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap() as u64,
            close: 4_000.0,
            open_interest: 100.0,
        });
        model
            .complete_bar(CompletedBar {
                timestamp_ns: datetime("20260810 14:40:00")
                    .and_utc()
                    .timestamp_nanos_opt()
                    .unwrap() as u64,
                close: 4_001.0,
                open_interest: 101.0,
            })
            .unwrap();
        assert_eq!(model.pending_entry.as_ref().unwrap().entry_row, 221);
        assert_eq!(model.pending_entry.as_ref().unwrap().exit_row, 240);

        let mut tail = Lab0344Model::new(&config());
        tail.action_day = Some("20260810".into());
        tail.completed_rows = 220;
        tail.previous = Some(CompletedBar {
            timestamp_ns: datetime("20260810 14:40:00")
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap() as u64,
            close: 4_000.0,
            open_interest: 100.0,
        });
        tail.complete_bar(CompletedBar {
            timestamp_ns: datetime("20260810 14:41:00")
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap() as u64,
            close: 4_001.0,
            open_interest: 101.0,
        })
        .unwrap();
        assert!(tail.pending_entry.is_none());
        assert!(!tail.signal_seen);
    }

    #[test]
    fn action_day_is_the_natural_day_boundary_not_ctp_trading_day() {
        let mut model = Lab0344Model::new(&config());
        let mut night = tick_at(datetime("20260809 21:00:00"), 4_000.0, 100.0);
        night.trading_day = "20260810".into();
        model.ingest(night).unwrap();
        assert_eq!(model.action_day.as_deref(), Some("20260809"));

        let mut morning = tick_at(datetime("20260810 09:30:00"), 4_001.0, 101.0);
        morning.trading_day = "20260810".into();
        model.ingest(morning).unwrap();
        assert_eq!(model.action_day.as_deref(), Some("20260810"));
        assert_eq!(model.completed_rows, 0);
    }

    #[test]
    fn rejects_out_of_order_ticks_and_clears_every_state_after_a_gap() {
        let mut portfolio = Portfolio::default();
        portfolio.insert("IF2609".into(), Lab0344Model::new(&config()));
        portfolio.ingest(session_tick(1, 4_001.0, 100.0)).unwrap();
        assert!(portfolio.ingest(session_tick(0, 4_000.0, 100.0)).is_err());
        portfolio.clear("IF2609");
        let model = portfolio.models.get("IF2609").unwrap();
        assert!(model.current.is_none());
        assert!(model.previous.is_none());
        assert!(model.position.is_none());
        assert!(model.pending_entry.is_none());
        assert_eq!(model.completed_rows, 0);
        assert!(!model.signal_seen);
    }

    #[test]
    fn malformed_market_values_are_rejected_before_inference() {
        let mut model = Lab0344Model::new(&config());
        let mut invalid = session_tick(0, 4_000.0, 1.0);
        invalid.open_interest = f64::NAN;
        assert!(model.ingest(invalid).is_err());
    }

    #[test]
    fn hot_path_p99_is_below_one_hundred_milliseconds() {
        let mut performance_config = config();
        performance_config.session_bar_count = 3_000;
        let mut model = Lab0344Model::new(&performance_config);
        let mut samples = Vec::new();
        let start = datetime("20260810 09:30:00");
        for row in 0..2_000 {
            let started = Instant::now();
            model
                .ingest(tick_at(
                    start + Duration::minutes(row as i64),
                    4_000.0 + (row % 3) as f64,
                    row as f64 + 1.0,
                ))
                .unwrap();
            samples.push(started.elapsed());
        }
        samples.sort_unstable();
        let p99 = samples[samples.len() * 99 / 100];
        assert!(p99.as_millis() < 100, "p99={p99:?}");
    }

    #[test]
    fn historical_datetime_keeps_the_parquet_natural_date() {
        let timestamp_ns = datetime("20260810 20:30:00")
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap() as u64;
        assert_eq!(super::history_day(timestamp_ns).unwrap(), "20260810");
    }

    #[test]
    fn short_target_keeps_its_signed_contract_count() {
        let mut model = Lab0344Model::new(&config());
        model.position = Some(super::OpenPosition {
            side: Side::Short,
            entry_price: 4_000.0,
            exit_row: 80,
        });
        model.latest_price = Some(3_990.0);
        model.exchange_id = "CFFEX".into();
        assert_eq!(model.target().unwrap().unwrap().contracts, -2);
    }
}

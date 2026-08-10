use std::collections::{BTreeMap, BTreeSet, VecDeque};

use bullet_data::HistoryBar;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Timelike, Utc};
use serde::Deserialize;

use crate::config::InstrumentConfig;

const NS_PER_MINUTE: u64 = 60_000_000_000;
const ROUND_TRIP_COST: f64 = 0.000_246;
const BASE_WEIGHT: f64 = 0.70;
const ADDON_WEIGHT: f64 = 0.30;
const OVERLAY_WEIGHT: f64 = 0.50;
const SOTA_SYMBOLS: [&str; 2] = ["IC8888", "IH8888"];

/// CTPD `event: tick` payload. Volume and turnover are required because
/// lab0334's VWAP, money z-score and IM overlay depend on them.
#[derive(Clone, Debug, Deserialize)]
pub struct CtpdTick {
    pub instrument_id: String,
    pub exchange_id: String,
    pub trading_day: String,
    pub action_day: String,
    pub update_time: String,
    pub update_millisec: i32,
    pub last_price: f64,
    pub volume: f64,
    pub turnover: f64,
    pub open_interest: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Side {
    Long,
    Short,
}

impl Side {
    fn sign(self) -> f64 {
        match self {
            Self::Long => 1.0,
            Self::Short => -1.0,
        }
    }

    fn raw_return(self, entry: f64, exit: f64) -> f64 {
        match self {
            Self::Long => exit / entry - 1.0,
            Self::Short => entry / exit - 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TargetPosition {
    pub symbol: String,
    pub target_instrument_id: String,
    pub exchange_id: String,
    pub contracts: f64,
    pub entry_price: f64,
    pub latest_price: f64,
    pub multiplier: f64,
    pub updated_at_ms: i64,
}

#[derive(Clone, Debug)]
struct MarketBar {
    timestamp_ns: u64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    money: f64,
}

impl From<&HistoryBar> for MarketBar {
    fn from(bar: &HistoryBar) -> Self {
        Self {
            timestamp_ns: bar.timestamp_ns,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
            money: bar.money,
        }
    }
}

#[derive(Clone, Debug)]
struct OpenBar {
    endpoint_ns: u64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    money: f64,
}

#[derive(Clone, Debug)]
struct Indicator {
    bar: MarketBar,
    day: NaiveDate,
    fast_ma: Option<f64>,
    slow_ma: Option<f64>,
    trend_ma: Option<f64>,
    signed_ma_gap_bps: Option<f64>,
    late_momentum_bps: Option<f64>,
    ret15_bps: Option<f64>,
    ret5_bps: Option<f64>,
    ret30_bps: Option<f64>,
    ret60_bps: Option<f64>,
    ret120_bps: Option<f64>,
    money_z30: Option<f64>,
    atr_z: Option<f64>,
    price_vs_vwap_bps: Option<f64>,
    range_pos: f64,
    day_open: f64,
    day_ret_bps: f64,
    signal: Option<Side>,
}

#[derive(Default)]
struct Indicators {
    closes: VecDeque<f64>,
    trend_mas: VecDeque<f64>,
    monies: VecDeque<f64>,
    true_ranges: VecDeque<f64>,
    atr_values: VecDeque<f64>,
    atr_sum: f64,
    atr_sum_squares: f64,
    last_close: Option<f64>,
    active_day: Option<NaiveDate>,
    day_volume: f64,
    day_weighted_close: f64,
    day_high: f64,
    day_low: f64,
    day_open: f64,
    previous_fast: Option<f64>,
    previous_slow: Option<f64>,
}

impl Indicators {
    fn reset_day(&mut self, bar: &MarketBar, day: NaiveDate) {
        self.active_day = Some(day);
        self.day_volume = 0.0;
        self.day_weighted_close = 0.0;
        self.day_high = bar.high;
        self.day_low = bar.low;
        self.day_open = bar.open;
    }

    fn push(&mut self, bar: MarketBar) -> Result<Indicator, String> {
        let day = market_day(bar.timestamp_ns)?;
        if self.active_day != Some(day) {
            self.reset_day(&bar, day);
        }
        let prior_money = self
            .monies
            .iter()
            .rev()
            .take(30)
            .copied()
            .collect::<Vec<_>>();
        let money_z30 = z_score(bar.money, &prior_money, 15);

        let prior_close = self.last_close;
        let true_range = match prior_close {
            Some(previous) => (bar.high - bar.low)
                .max((bar.high - previous).abs())
                .max((bar.low - previous).abs()),
            None => bar.high - bar.low,
        };
        self.true_ranges.push_back(true_range);
        trim(&mut self.true_ranges, 30);
        let atr30 = (self.true_ranges.len() == 30)
            .then(|| self.true_ranges.iter().sum::<f64>() / 30.0 / bar.close * 10_000.0);
        let atr_z = atr30.and_then(|value| {
            z_score_from_moments(
                self.atr_values.len(),
                self.atr_sum,
                self.atr_sum_squares,
                value,
                1_000,
            )
        });
        if let Some(value) = atr30 {
            self.atr_values.push_back(value);
            self.atr_sum += value;
            self.atr_sum_squares += value * value;
            while self.atr_values.len() > 5_000 {
                let removed = self.atr_values.pop_front().expect("non-empty ATR history");
                self.atr_sum -= removed;
                self.atr_sum_squares -= removed * removed;
            }
        }

        self.day_volume += bar.volume;
        self.day_weighted_close += bar.close * bar.volume;
        self.day_high = self.day_high.max(bar.high);
        self.day_low = self.day_low.min(bar.low);
        let vwap = (self.day_volume > 0.0).then(|| self.day_weighted_close / self.day_volume);
        let range_pos = if self.day_high > self.day_low {
            (bar.close - self.day_low) / (self.day_high - self.day_low)
        } else {
            0.5
        };

        self.closes.push_back(bar.close);
        trim(&mut self.closes, 300);
        let fast_ma = mean_tail(&self.closes, 20);
        let slow_ma = mean_tail(&self.closes, 60);
        let trend_ma = mean_tail(&self.closes, 240);
        let trend_slope_bps = match (
            trend_ma,
            self.trend_mas.get(self.trend_mas.len().saturating_sub(60)),
        ) {
            (Some(current), Some(prior)) => Some((current - prior) / bar.close * 10_000.0),
            _ => None,
        };
        if let Some(value) = trend_ma {
            self.trend_mas.push_back(value);
            trim(&mut self.trend_mas, 60);
        }
        let ma_gap_bps = match (fast_ma, slow_ma) {
            (Some(fast), Some(slow)) => Some((fast - slow).abs() / bar.close * 10_000.0),
            _ => None,
        };
        let signed_ma_gap_bps = match (fast_ma, slow_ma) {
            (Some(fast), Some(slow)) => Some((fast / slow - 1.0) * 10_000.0),
            _ => None,
        };
        let late_momentum_bps = return_bps(&self.closes, bar.close, 30);
        let ret5_bps = return_bps(&self.closes, bar.close, 5);
        let ret15_bps = return_bps(&self.closes, bar.close, 15);
        let ret30_bps = return_bps(&self.closes, bar.close, 30);
        let ret60_bps = return_bps(&self.closes, bar.close, 60);
        let ret120_bps = return_bps(&self.closes, bar.close, 120);
        let price_vs_vwap_bps = vwap.map(|value| (bar.close / value - 1.0) * 10_000.0);
        let signal = match (
            fast_ma,
            slow_ma,
            self.previous_fast,
            self.previous_slow,
            trend_ma,
            trend_slope_bps,
            ma_gap_bps,
        ) {
            (
                Some(fast),
                Some(slow),
                Some(prev_fast),
                Some(prev_slow),
                Some(trend),
                Some(slope),
                Some(gap),
            ) if fast > slow
                && prev_fast <= prev_slow
                && bar.close > trend
                && slope >= 10.0
                && gap >= 2.0 =>
            {
                Some(Side::Long)
            }
            (
                Some(fast),
                Some(slow),
                Some(prev_fast),
                Some(prev_slow),
                Some(trend),
                Some(slope),
                Some(gap),
            ) if fast < slow
                && prev_fast >= prev_slow
                && bar.close < trend
                && slope <= -10.0
                && gap >= 2.0 =>
            {
                Some(Side::Short)
            }
            _ => None,
        };
        self.previous_fast = fast_ma;
        self.previous_slow = slow_ma;
        self.monies.push_back(bar.money);
        trim(&mut self.monies, 30);
        self.last_close = Some(bar.close);
        let day_ret_bps = (bar.close / self.day_open - 1.0) * 10_000.0;
        Ok(Indicator {
            bar,
            day,
            fast_ma,
            slow_ma,
            trend_ma,
            signed_ma_gap_bps,
            late_momentum_bps,
            ret15_bps,
            ret5_bps,
            ret30_bps,
            ret60_bps,
            ret120_bps,
            money_z30,
            atr_z,
            price_vs_vwap_bps,
            range_pos,
            day_open: self.day_open,
            day_ret_bps,
            signal,
        })
    }
}

#[derive(Clone, Debug)]
enum ExitPlan {
    EndOfDay,
    AtOrAfter(u64),
    Conditional,
}

#[derive(Clone, Debug)]
struct Candidate {
    id: String,
    symbol: String,
    side: Side,
    family: &'static str,
    policy: &'static str,
    weight: f64,
    signal_time_ns: u64,
    entry_time_ns: u64,
    entry_price: f64,
    exit_plan: ExitPlan,
    ret30_signed: f64,
    ret60_signed: f64,
    vwap_signed: f64,
    trend_distance_bps: f64,
}

#[derive(Clone, Debug)]
struct Draft {
    side: Side,
    family: &'static str,
    policy: &'static str,
    weight: f64,
    signal_time_ns: u64,
    exit_plan: ExitPlan,
    ret30_signed: f64,
    ret60_signed: f64,
    vwap_signed: f64,
    trend_distance_bps: f64,
    virtual_kind: VirtualKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VirtualKind {
    Base,
    Addon,
    Overlay,
}

#[derive(Clone, Debug)]
struct VirtualLeg {
    candidate: Candidate,
    exit_next_open: bool,
    delayed_bad_at_row: Option<usize>,
}

#[derive(Clone, Debug)]
enum ModelEvent {
    Candidate(Candidate),
    Label {
        candidate: Candidate,
        exit_price: f64,
        at_ns: u64,
    },
    Price,
}

pub(crate) struct InstrumentModel {
    config: InstrumentConfig,
    current: Option<OpenBar>,
    indicators: Indicators,
    last_indicator: Option<Indicator>,
    latest_price: Option<f64>,
    latest_exchange: String,
    latest_at_ns: Option<u64>,
    bar_opens: VecDeque<(u64, f64)>,
    last_tick_ns: Option<u64>,
    last_cumulative_volume: Option<f64>,
    last_cumulative_turnover: Option<f64>,
    day_rows: usize,
    base_finished: bool,
    base: Option<VirtualLeg>,
    addon: Option<VirtualLeg>,
    overlays: Vec<VirtualLeg>,
    pending: Vec<Draft>,
    overlay_seen: bool,
    recovery_seen: bool,
    im_seen: bool,
    early_seen: bool,
    if_seen: bool,
    candidate_sequence: u64,
}

impl InstrumentModel {
    fn new(config: &InstrumentConfig) -> Self {
        Self {
            config: config.clone(),
            current: None,
            indicators: Indicators::default(),
            last_indicator: None,
            latest_price: None,
            latest_exchange: String::new(),
            latest_at_ns: None,
            bar_opens: VecDeque::new(),
            last_tick_ns: None,
            last_cumulative_volume: None,
            last_cumulative_turnover: None,
            day_rows: 0,
            base_finished: false,
            base: None,
            addon: None,
            overlays: Vec::new(),
            pending: Vec::new(),
            overlay_seen: false,
            recovery_seen: false,
            im_seen: false,
            early_seen: false,
            if_seen: false,
            candidate_sequence: 0,
        }
    }

    fn ingest_history(&mut self, bar: &HistoryBar) -> Result<Vec<ModelEvent>, String> {
        let mut events = self.begin_bar(bar.timestamp_ns, bar.open, "CFFEX")?;
        events.extend(self.complete_bar(MarketBar::from(bar))?);
        Ok(events)
    }

    fn ingest_tick(&mut self, tick: CtpdTick) -> Result<Vec<ModelEvent>, String> {
        if tick.instrument_id != self.config.market_instrument_id {
            return Err(format!("unexpected CTPD instrument {}", tick.instrument_id));
        }
        if tick.trading_day.trim().is_empty()
            || !tick.last_price.is_finite()
            || tick.last_price <= 0.0
            || !tick.volume.is_finite()
            || !tick.turnover.is_finite()
            || !tick.open_interest.is_finite()
            || tick.volume < 0.0
            || tick.turnover < 0.0
            || tick.open_interest < 0.0
        {
            return Err("CTPD tick has invalid market values".into());
        }
        let tick_ns = tick_timestamp_ns(&tick)?;
        let endpoint = bar_endpoint_ns(tick_ns)?;
        if self
            .last_indicator
            .as_ref()
            .is_some_and(|bar| endpoint <= bar.bar.timestamp_ns)
        {
            return Ok(Vec::new());
        }
        if self.last_tick_ns.is_some_and(|prior| tick_ns < prior) {
            return Err("CTPD tick is out of order".into());
        }
        let volume_delta = cumulative_delta(tick.volume, self.last_cumulative_volume);
        let money_delta = cumulative_delta(tick.turnover, self.last_cumulative_turnover);
        self.last_cumulative_volume = Some(tick.volume);
        self.last_cumulative_turnover = Some(tick.turnover);
        self.last_tick_ns = Some(tick_ns);
        self.latest_price = Some(tick.last_price);
        self.latest_exchange = tick.exchange_id.clone();
        self.latest_at_ns = Some(tick_ns);
        let mut events = vec![ModelEvent::Price];
        match self.current.as_mut() {
            None => {
                events.extend(self.begin_live_bar(
                    endpoint,
                    tick.last_price,
                    volume_delta,
                    money_delta,
                    &tick.exchange_id,
                )?);
            }
            Some(current) if current.endpoint_ns == endpoint => {
                current.close = tick.last_price;
                current.high = current.high.max(tick.last_price);
                current.low = current.low.min(tick.last_price);
                current.volume += volume_delta;
                current.money += money_delta;
            }
            Some(current) if current.endpoint_ns < endpoint => {
                let completed = MarketBar {
                    timestamp_ns: current.endpoint_ns,
                    open: current.open,
                    high: current.high,
                    low: current.low,
                    close: current.close,
                    volume: current.volume,
                    money: current.money,
                };
                self.current = None;
                events.extend(self.complete_bar(completed)?);
                events.extend(self.begin_live_bar(
                    endpoint,
                    tick.last_price,
                    volume_delta,
                    money_delta,
                    &tick.exchange_id,
                )?);
            }
            Some(_) => return Err("CTPD tick moved before current minute".into()),
        }
        Ok(events)
    }

    fn begin_live_bar(
        &mut self,
        endpoint_ns: u64,
        price: f64,
        volume: f64,
        money: f64,
        exchange: &str,
    ) -> Result<Vec<ModelEvent>, String> {
        let events = self.begin_bar(endpoint_ns, price, exchange)?;
        self.current = Some(OpenBar {
            endpoint_ns,
            open: price,
            high: price,
            low: price,
            close: price,
            volume,
            money,
        });
        Ok(events)
    }

    fn begin_bar(
        &mut self,
        endpoint_ns: u64,
        open: f64,
        exchange: &str,
    ) -> Result<Vec<ModelEvent>, String> {
        let day = market_day(endpoint_ns)?;
        if self
            .last_indicator
            .as_ref()
            .is_some_and(|previous| previous.day != day)
        {
            self.clear_day();
            self.last_indicator = None;
        }
        let mut events = Vec::new();
        self.latest_price = Some(open);
        self.latest_exchange = exchange.to_owned();
        self.latest_at_ns = Some(endpoint_ns);
        self.bar_opens.push_back((endpoint_ns, open));
        while self.bar_opens.len() > 300 {
            self.bar_opens.pop_front();
        }
        events.push(ModelEvent::Price);
        self.resolve_due_exits(endpoint_ns, open, &mut events)?;
        self.activate_base_signal(endpoint_ns, open, &mut events)?;
        let pending = std::mem::take(&mut self.pending);
        for draft in pending {
            self.activate_draft(draft, endpoint_ns, open, &mut events)?;
        }
        if is_session_end(endpoint_ns)? {
            self.resolve_all_eod(endpoint_ns, open, &mut events);
        }
        Ok(events)
    }

    fn complete_bar(&mut self, bar: MarketBar) -> Result<Vec<ModelEvent>, String> {
        self.day_rows += 1;
        let indicator = self.indicators.push(bar)?;
        let events = Vec::new();
        self.evaluate_base_exit(&indicator);
        self.maybe_schedule_addon(&indicator);
        self.maybe_schedule_overlays(&indicator)?;
        self.last_indicator = Some(indicator);
        Ok(events)
    }

    fn activate_base_signal(
        &mut self,
        at_ns: u64,
        open: f64,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), String> {
        let Some(previous) = self.last_indicator.as_ref() else {
            return Ok(());
        };
        if !SOTA_SYMBOLS.contains(&self.config.symbol.as_str())
            || self.base.is_some()
            || self.base_finished
            || market_hour(at_ns)? == 10
        {
            return Ok(());
        }
        let Some(side) = previous.signal else {
            return Ok(());
        };
        let distance = match (previous.trend_ma, previous.bar.close) {
            (Some(trend), close) if trend > 0.0 => match side {
                Side::Long => (close / trend - 1.0) * 10_000.0,
                Side::Short => (trend / close - 1.0) * 10_000.0,
            },
            _ => 0.0,
        };
        let draft = Draft {
            side,
            family: "base",
            policy: "main_conditional_stop_late_state_or_eod_open",
            weight: 1.0,
            signal_time_ns: previous.bar.timestamp_ns,
            exit_plan: ExitPlan::Conditional,
            ret30_signed: 0.0,
            ret60_signed: 0.0,
            vwap_signed: 0.0,
            trend_distance_bps: distance,
            virtual_kind: VirtualKind::Base,
        };
        self.activate_draft(draft, at_ns, open, events)
    }

    fn activate_draft(
        &mut self,
        draft: Draft,
        at_ns: u64,
        open: f64,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), String> {
        self.candidate_sequence += 1;
        let candidate = Candidate {
            id: format!(
                "{}-{}-{}",
                self.config.symbol, at_ns, self.candidate_sequence
            ),
            symbol: self.config.symbol.clone(),
            side: draft.side,
            family: draft.family,
            policy: draft.policy,
            weight: draft.weight,
            signal_time_ns: draft.signal_time_ns,
            entry_time_ns: at_ns,
            entry_price: open,
            exit_plan: draft.exit_plan,
            ret30_signed: draft.ret30_signed,
            ret60_signed: draft.ret60_signed,
            vwap_signed: draft.vwap_signed,
            trend_distance_bps: draft.trend_distance_bps,
        };
        let leg = VirtualLeg {
            candidate: candidate.clone(),
            exit_next_open: false,
            delayed_bad_at_row: None,
        };
        match draft.virtual_kind {
            VirtualKind::Base => self.base = Some(leg),
            VirtualKind::Addon => self.addon = Some(leg),
            VirtualKind::Overlay => self.overlays.push(leg),
        }
        events.push(ModelEvent::Candidate(candidate));
        Ok(())
    }

    fn resolve_due_exits(
        &mut self,
        at_ns: u64,
        open: f64,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), String> {
        if self.base.as_ref().is_some_and(|leg| leg.exit_next_open) {
            let base = self.base.take().expect("checked base exists");
            self.base_finished = true;
            label_event(base.candidate, open, at_ns, events);
            if let Some(addon) = self.addon.take() {
                label_event(addon.candidate, open, at_ns, events);
            }
        }
        let mut pending = Vec::new();
        for leg in self.overlays.drain(..) {
            let due =
                matches!(leg.candidate.exit_plan, ExitPlan::AtOrAfter(target) if at_ns >= target);
            if due {
                label_event(leg.candidate, open, at_ns, events);
            } else {
                pending.push(leg);
            }
        }
        self.overlays = pending;
        Ok(())
    }

    fn resolve_all_eod(&mut self, at_ns: u64, open: f64, events: &mut Vec<ModelEvent>) {
        if let Some(base) = self.base.take() {
            label_event(base.candidate, open, at_ns, events);
            self.base_finished = true;
        }
        if let Some(addon) = self.addon.take() {
            label_event(addon.candidate, open, at_ns, events);
        }
        for overlay in self.overlays.drain(..) {
            label_event(overlay.candidate, open, at_ns, events);
        }
    }

    fn evaluate_base_exit(&mut self, row: &Indicator) {
        let Some(base) = self.base.as_mut() else {
            return;
        };
        if row.bar.timestamp_ns <= base.candidate.entry_time_ns {
            return;
        }
        let unrealized = base
            .candidate
            .side
            .raw_return(base.candidate.entry_price, row.bar.close)
            * 10_000.0;
        if self.config.symbol == "IC8888" && unrealized <= -40.0 {
            base.exit_next_open = true;
            return;
        }
        let bars_to_close = self.config.session_bar_count.saturating_sub(self.day_rows);
        let bad = bad_late_state(row, &base.candidate, bars_to_close);
        if !bad {
            return;
        }
        if self.config.symbol == "IC8888"
            && base.candidate.side == Side::Short
            && (16..=30).contains(&bars_to_close)
        {
            match base.delayed_bad_at_row {
                None => {
                    base.delayed_bad_at_row = Some(self.day_rows + 5);
                    return;
                }
                Some(confirm) if self.day_rows < confirm => return,
                Some(_) => base.delayed_bad_at_row = None,
            }
        }
        base.exit_next_open = true;
    }

    fn maybe_schedule_addon(&mut self, row: &Indicator) {
        let Some(base) = self.base.as_ref() else {
            return;
        };
        if self.addon.is_some()
            || base.candidate.weight != BASE_WEIGHT
            || row.bar.timestamp_ns <= base.candidate.entry_time_ns
        {
            return;
        }
        let (Some(ret30), Some(ret60), Some(gap), Some(vwap)) = (
            row.ret30_bps,
            row.ret60_bps,
            row.signed_ma_gap_bps,
            row.price_vs_vwap_bps,
        ) else {
            return;
        };
        let side = base.candidate.side;
        let signed = |value: f64| value * side.sign();
        if signed(ret30) >= 25.0
            && signed(ret60) >= 20.0
            && signed(gap) >= 1.0
            && signed(vwap) >= 40.0
        {
            self.pending.push(Draft {
                side,
                family: "addon",
                policy: "follow_main_conditional_or_eod_open",
                weight: ADDON_WEIGHT,
                signal_time_ns: row.bar.timestamp_ns,
                exit_plan: ExitPlan::Conditional,
                ret30_signed: signed(ret30),
                ret60_signed: signed(ret60),
                vwap_signed: signed(vwap),
                trend_distance_bps: 0.0,
                virtual_kind: VirtualKind::Addon,
            });
        }
    }

    fn maybe_schedule_overlays(&mut self, row: &Indicator) -> Result<(), String> {
        let time = time_hms(row.bar.timestamp_ns)?;
        if !self.early_seen
            && matches!(self.config.symbol.as_str(), "IC8888" | "IM8888")
            && (91_600..10_29_00).contains(&time)
            && let (Some(ret5), Some(vwap)) = (row.ret5_bps, row.price_vs_vwap_bps)
            && row.day_ret_bps >= 80.0
            && ret5 <= -5.0
            && (0.0..=30.0).contains(&vwap)
            && row.range_pos >= 0.55
        {
            let ic_entry_minute_blocked = self.config.symbol == "IC8888"
                && ((vwap >= 20.0)
                    || (51..=59).contains(
                        &market_datetime(row.bar.timestamp_ns + NS_PER_MINUTE)?.minute(),
                    ));
            if ic_entry_minute_blocked {
                // The lab gives IC's first low-edge rejection veto over later
                // same-day early-pullback candidates.
                self.early_seen = true;
            } else {
                self.early_seen = true;
                self.pending.push(Draft {
                    side: Side::Long,
                    family: "idle_overlay",
                    policy: "early_pullback_h45_open",
                    weight: 0.30,
                    signal_time_ns: row.bar.timestamp_ns,
                    exit_plan: ExitPlan::AtOrAfter(row.bar.timestamp_ns + 46 * NS_PER_MINUTE),
                    ret30_signed: ret5,
                    ret60_signed: row.ret30_bps.unwrap_or(0.0),
                    vwap_signed: vwap,
                    trend_distance_bps: 0.0,
                    virtual_kind: VirtualKind::Overlay,
                });
            }
        }
        if !self.overlay_seen
            && (133_000..144_500).contains(&time)
            && let (Some(ret30), Some(ret60), Some(vwap)) =
                (row.ret30_bps, row.ret60_bps, row.price_vs_vwap_bps)
        {
            let side = if ret30 >= 35.0 && ret60 >= 40.0 && vwap >= 70.0 {
                Some(Side::Long)
            } else if ret30 <= -35.0 && ret60 <= -40.0 && vwap <= -70.0 {
                Some(Side::Short)
            } else {
                None
            };
            if let Some(side) = side {
                let signed_range = if side == Side::Long {
                    row.range_pos
                } else {
                    1.0 - row.range_pos
                };
                if signed_range >= 0.8
                    && !(matches!(self.config.symbol.as_str(), "IF8888" | "IM8888")
                        && ret30 * side.sign() >= 80.0)
                {
                    self.overlay_seen = true;
                    self.pending.push(Draft {
                        side,
                        family: "idle_overlay",
                        policy: "ordinary_idle_eod_open",
                        weight: OVERLAY_WEIGHT,
                        signal_time_ns: row.bar.timestamp_ns,
                        exit_plan: ExitPlan::EndOfDay,
                        ret30_signed: ret30 * side.sign(),
                        ret60_signed: ret60 * side.sign(),
                        vwap_signed: vwap * side.sign(),
                        trend_distance_bps: 0.0,
                        virtual_kind: VirtualKind::Overlay,
                    });
                }
            }
        }
        if !self.recovery_seen
            && (103_000..140_000).contains(&time)
            && let (Some(ret15), Some(ret60), Some(ret120), Some(vwap)) = (
                row.ret15_bps,
                row.ret60_bps,
                row.ret120_bps,
                row.price_vs_vwap_bps,
            )
        {
            let side = if ret120 >= 120.0 && (0.0..=35.0).contains(&vwap) && ret15 >= 10.0 {
                Some(Side::Long)
            } else if ret120 <= -120.0 && (-35.0..=0.0).contains(&vwap) && ret15 <= -10.0 {
                Some(Side::Short)
            } else {
                None
            };
            if let Some(side) = side {
                self.recovery_seen = true;
                self.pending.push(Draft {
                    side,
                    family: "idle_overlay",
                    policy: "trend_recovery_h45_open",
                    weight: 0.25,
                    signal_time_ns: row.bar.timestamp_ns,
                    exit_plan: ExitPlan::AtOrAfter(row.bar.timestamp_ns + 46 * NS_PER_MINUTE),
                    ret30_signed: ret15 * side.sign(),
                    ret60_signed: ret60 * side.sign(),
                    vwap_signed: vwap * side.sign(),
                    trend_distance_bps: 0.0,
                    virtual_kind: VirtualKind::Overlay,
                });
            }
        }
        if self.config.symbol == "IM8888" && !self.im_seen {
            let day_ret = (row.bar.close / row.day_open - 1.0) * 10_000.0;
            if day_ret.abs() >= 120.0 {
                let side = if day_ret > 0.0 {
                    Side::Long
                } else {
                    Side::Short
                };
                let state_ok = if day_ret.abs() < 140.0 {
                    side == Side::Short
                        || (row.atr_z.is_some_and(|z| z < 0.5)
                            && row.price_vs_vwap_bps.is_some_and(|value| value < 140.0))
                } else {
                    row.atr_z
                        .is_some_and(|z| z < if side == Side::Long { 0.5 } else { 1.5 })
                };
                if state_ok && time <= 141_400 {
                    self.im_seen = true;
                    self.pending.push(Draft {
                        side,
                        family: "idle_overlay",
                        policy: "im_day120_140_h45_open",
                        weight: 0.30,
                        signal_time_ns: row.bar.timestamp_ns,
                        exit_plan: ExitPlan::AtOrAfter(row.bar.timestamp_ns + 46 * NS_PER_MINUTE),
                        ret30_signed: 0.0,
                        ret60_signed: 0.0,
                        vwap_signed: 0.0,
                        trend_distance_bps: 0.0,
                        virtual_kind: VirtualKind::Overlay,
                    });
                }
            }
        }
        if self.config.symbol == "IF8888"
            && !self.if_seen
            && (94_600..145_000).contains(&time)
            && market_hour(row.bar.timestamp_ns)? == 10
            && let (Some(ret15), Some(vwap), Some(money_z)) =
                (row.ret15_bps, row.price_vs_vwap_bps, row.money_z30)
            && (80.0..=120.0).contains(&vwap)
            && (20.0..=120.0).contains(&ret15)
            && money_z >= 1.0
        {
            self.if_seen = true;
            self.pending.push(Draft {
                side: Side::Long,
                family: "idle_overlay",
                policy: "if_vwap_confirm_eod_open",
                weight: 0.30,
                signal_time_ns: row.bar.timestamp_ns,
                exit_plan: ExitPlan::EndOfDay,
                ret30_signed: ret15,
                ret60_signed: row.ret30_bps.unwrap_or(0.0),
                vwap_signed: vwap,
                trend_distance_bps: 0.0,
                virtual_kind: VirtualKind::Overlay,
            });
        }
        Ok(())
    }

    fn clear_day(&mut self) {
        self.current = None;
        self.day_rows = 0;
        self.base_finished = false;
        self.base = None;
        self.addon = None;
        self.overlays.clear();
        self.pending.clear();
        self.overlay_seen = false;
        self.recovery_seen = false;
        self.im_seen = false;
        self.early_seen = false;
        self.if_seen = false;
        self.last_cumulative_volume = None;
        self.last_cumulative_turnover = None;
        self.bar_opens.clear();
    }

    fn clear_live_state(&mut self) {
        // The completed-bar state and virtual legs are still authoritative up
        // to `last_indicator`. A reconnect first recovers every missing closed
        // Kline against that state; clearing them would make an intraday base
        // position impossible to reconstruct after a transient SSE outage.
        self.current = None;
        self.latest_price = None;
        self.latest_exchange.clear();
        self.latest_at_ns = None;
        self.last_tick_ns = None;
        self.last_cumulative_volume = None;
        self.last_cumulative_turnover = None;
    }

    fn set_candidate_weight(&mut self, candidate_id: &str, weight: f64) {
        let apply = |leg: &mut VirtualLeg| {
            if leg.candidate.id == candidate_id {
                leg.candidate.weight = weight;
            }
        };
        if let Some(base) = self.base.as_mut() {
            apply(base);
        }
        if let Some(addon) = self.addon.as_mut() {
            apply(addon);
        }
        for overlay in &mut self.overlays {
            apply(overlay);
        }
    }

    fn cancel_candidate(&mut self, candidate_id: &str) {
        if self
            .base
            .as_ref()
            .is_some_and(|leg| leg.candidate.id == candidate_id)
        {
            self.base = None;
            self.base_finished = true;
        }
        if self
            .addon
            .as_ref()
            .is_some_and(|leg| leg.candidate.id == candidate_id)
        {
            self.addon = None;
        }
        self.overlays.retain(|leg| leg.candidate.id != candidate_id);
    }

    fn open_at(&self, timestamp_ns: u64) -> Option<f64> {
        self.bar_opens
            .iter()
            .find_map(|(time, price)| (*time == timestamp_ns).then_some(*price))
    }
}

#[derive(Default)]
struct Arbitrator {
    active: BTreeMap<String, Candidate>,
    histories: BTreeMap<String, Vec<(u64, f64)>>,
    idle_histories: BTreeMap<String, Vec<f64>>,
    pending_labels: BTreeMap<u64, Vec<(Candidate, f64)>>,
}

impl Arbitrator {
    fn resolve_label(&mut self, candidate: Candidate, exit_price: f64, at_ns: u64) {
        self.active.remove(&candidate.id);
        let trade_return =
            candidate.side.raw_return(candidate.entry_price, exit_price) - ROUND_TRIP_COST;
        self.pending_labels
            .entry(at_ns)
            .or_default()
            .push((candidate, trade_return));
    }

    fn decide(
        &mut self,
        candidate: Candidate,
        replacement_prices: &BTreeMap<String, f64>,
    ) -> Option<(Candidate, f64)> {
        self.mature_before(candidate.entry_time_ns);
        let (prediction, _) = self.predict(&candidate);
        if prediction <= 0.0
            || self.recovery_gate(&candidate, prediction)
            || self.idle_gate(&candidate)
        {
            return None;
        }
        let used = self.active.values().map(portfolio_weight).sum::<f64>();
        let same_symbol = self
            .active
            .values()
            .filter(|active| active.symbol == candidate.symbol)
            .map(|active| active.id.clone())
            .collect::<Vec<_>>();
        let candidate_weight = portfolio_weight(&candidate);
        if same_symbol.is_empty() && used + candidate_weight <= 1.0 + 1e-12 {
            self.active.insert(candidate.id.clone(), candidate);
            return None;
        }
        let pool = if same_symbol.is_empty() {
            self.active
                .values()
                .map(|active| active.id.clone())
                .collect::<Vec<_>>()
        } else {
            same_symbol
        };
        let incumbent_id = pool.into_iter().min_by(|left, right| {
            let left_score = self
                .active
                .get(left)
                .map(|value| self.predict(value).0)
                .unwrap_or(0.0);
            let right_score = self
                .active
                .get(right)
                .map(|value| self.predict(value).0)
                .unwrap_or(0.0);
            left_score.total_cmp(&right_score)
        })?;
        let incumbent = self.active.get(&incumbent_id).expect("incumbent exists");
        let incumbent_prediction = self.predict(incumbent).0;
        let after_replace = used - portfolio_weight(incumbent) + candidate_weight;
        let idle_release = candidate.policy == "ordinary_idle_eod_open"
            && candidate.side == Side::Long
            && self.active.len() >= 2
            && candidate.vwap_signed >= 80.0
            && candidate.ret60_signed <= 100.0
            && prediction >= 0.001
            && prediction >= incumbent_prediction;
        let margin = if idle_release { 0.0 } else { 0.002 };
        if after_replace <= 1.0 + 1e-12 && prediction >= incumbent_prediction + margin {
            let price = replacement_prices.get(&incumbent.symbol).copied()?;
            let incumbent = incumbent.clone();
            self.active.remove(&incumbent_id);
            self.active.insert(candidate.id.clone(), candidate);
            return Some((incumbent, price));
        }
        None
    }

    fn mature_before(&mut self, as_of_ns: u64) {
        let keys = self
            .pending_labels
            .range(..as_of_ns)
            .map(|(time, _)| *time)
            .collect::<Vec<_>>();
        for time in keys {
            let labels = self
                .pending_labels
                .remove(&time)
                .expect("pending label key exists");
            for (candidate, value) in labels {
                for key in candidate_keys(&candidate) {
                    self.histories.entry(key).or_default().push((time, value));
                }
                if candidate.policy == "ordinary_idle_eod_open" {
                    self.idle_histories
                        .entry(idle_key(&candidate))
                        .or_default()
                        .push(value * candidate.weight);
                }
            }
        }
    }

    fn predict(&self, candidate: &Candidate) -> (f64, usize) {
        for key in candidate_keys(candidate) {
            let values = self.histories.get(&key).map(Vec::as_slice).unwrap_or(&[]);
            if values.len() >= 20 || key == "global" {
                let recent = values
                    .iter()
                    .rev()
                    .take(250)
                    .map(|(_, value)| *value)
                    .collect::<Vec<_>>();
                if recent.is_empty() {
                    return (0.0, 0);
                }
                return (
                    recent.iter().sum::<f64>() / recent.len() as f64,
                    values.len(),
                );
            }
        }
        (0.0, 0)
    }

    fn recovery_gate(&self, candidate: &Candidate, prediction: f64) -> bool {
        candidate.policy == "trend_recovery_h45_open"
            && (0.0..=30.0).contains(&candidate.vwap_signed)
            && prediction <= 0.001
    }

    fn idle_gate(&self, candidate: &Candidate) -> bool {
        if candidate.policy != "ordinary_idle_eod_open"
            || !((80.0..=120.0).contains(&candidate.ret30_signed)
                || (100.0..=130.0).contains(&candidate.vwap_signed))
        {
            return false;
        }
        let values = self
            .idle_histories
            .get(&idle_key(candidate))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        values.len() >= 5
            && values.iter().rev().take(40).sum::<f64>() / (values.len().min(40) as f64) < 0.0
    }
}

#[derive(Default)]
pub struct Portfolio {
    models: BTreeMap<String, InstrumentModel>,
    symbols: BTreeMap<String, String>,
    arbitrator: Arbitrator,
    pending_candidates: BTreeMap<u64, Vec<Candidate>>,
    base_reserve_history: Vec<f64>,
    synchronized_markets: BTreeSet<String>,
    targets: BTreeMap<String, TargetPosition>,
}

impl Portfolio {
    pub fn insert(&mut self, market_instrument_id: String, model: InstrumentModel) {
        self.symbols
            .insert(model.config.symbol.clone(), market_instrument_id.clone());
        self.models.insert(market_instrument_id, model);
    }

    pub fn new_model(config: &InstrumentConfig) -> InstrumentModel {
        InstrumentModel::new(config)
    }

    pub fn ingest(&mut self, tick: CtpdTick) -> Result<(), String> {
        let id = tick.instrument_id.clone();
        let events = self
            .models
            .get_mut(&id)
            .ok_or_else(|| format!("no configured model for CTPD instrument {id}"))?
            .ingest_tick(tick)?;
        self.apply(events);
        Ok(())
    }

    pub fn ingest_history(
        &mut self,
        market_instrument_id: &str,
        bar: &HistoryBar,
    ) -> Result<(), String> {
        let events = self
            .models
            .get_mut(market_instrument_id)
            .ok_or_else(|| format!("no configured model for history {market_instrument_id}"))?
            .ingest_history(bar)?;
        self.apply(events);
        Ok(())
    }

    /// Kline recovery has an overlap with the Parquet splice and may be
    /// repeated after reconnect. Only strictly newer completed bars may alter
    /// the causal state.
    pub fn ingest_recovered_history(
        &mut self,
        market_instrument_id: &str,
        bar: &HistoryBar,
    ) -> Result<bool, String> {
        if self
            .last_bar_timestamp_ns(market_instrument_id)
            .is_some_and(|last| bar.timestamp_ns <= last)
        {
            return Ok(false);
        }
        self.ingest_history(market_instrument_id, bar)?;
        Ok(true)
    }

    pub fn mark_market_synchronized(&mut self, market_instrument_id: &str) -> Result<(), String> {
        if !self.models.contains_key(market_instrument_id) {
            return Err(format!(
                "no configured model for CTPD instrument {market_instrument_id}"
            ));
        }
        self.synchronized_markets
            .insert(market_instrument_id.to_owned());
        self.publish_targets();
        Ok(())
    }

    fn apply(&mut self, events: Vec<ModelEvent>) {
        for event in events {
            match event {
                ModelEvent::Candidate(candidate) => self
                    .pending_candidates
                    .entry(candidate.entry_time_ns)
                    .or_default()
                    .push(candidate),
                ModelEvent::Label {
                    candidate,
                    exit_price,
                    at_ns,
                } => self.arbitrator.resolve_label(candidate, exit_price, at_ns),
                ModelEvent::Price => {}
            }
        }
        self.flush_candidates();
        self.publish_targets();
    }

    fn flush_candidates(&mut self) {
        let Some(watermark) = self
            .models
            .values()
            .map(|model| model.latest_at_ns)
            .collect::<Option<Vec<_>>>()
            .and_then(|times| times.into_iter().min())
        else {
            return;
        };
        let due = self
            .pending_candidates
            .range(..=watermark)
            .map(|(time, _)| *time)
            .collect::<Vec<_>>();
        for time in due {
            let mut candidates = self
                .pending_candidates
                .remove(&time)
                .expect("pending candidate timestamp exists");
            candidates.sort_by(|left, right| {
                left.family
                    .cmp(right.family)
                    .then_with(|| left.symbol.cmp(&right.symbol))
                    .then_with(|| left.id.cmp(&right.id))
            });
            self.apply_base_dynamic_reserve(&mut candidates);
            for candidate in candidates {
                if !self.ordinary_idle_sync_blocked(&candidate) {
                    let entry_time = candidate.entry_time_ns;
                    let prices = self.open_prices_at(entry_time);
                    if let Some((incumbent, exit_price)) =
                        self.arbitrator.decide(candidate, &prices)
                    {
                        self.cancel_candidate(&incumbent.id);
                        self.arbitrator
                            .resolve_label(incumbent, exit_price, entry_time);
                    }
                }
            }
        }
    }

    fn apply_base_dynamic_reserve(&mut self, candidates: &mut [Candidate]) {
        let history_count = self.base_reserve_history.len();
        let threshold = (history_count >= 30).then(|| quantile(&self.base_reserve_history, 0.75));
        let mut observed = Vec::new();
        for candidate in candidates
            .iter_mut()
            .filter(|candidate| candidate.family == "base")
        {
            candidate.weight =
                if threshold.is_some_and(|value| candidate.trend_distance_bps >= value) {
                    BASE_WEIGHT
                } else {
                    1.0
                };
            if let Some(market_id) = self.symbols.get(&candidate.symbol)
                && let Some(model) = self.models.get_mut(market_id)
            {
                model.set_candidate_weight(&candidate.id, candidate.weight);
            }
            observed.push(candidate.trend_distance_bps);
        }
        self.base_reserve_history.extend(observed);
    }

    fn open_prices_at(&self, timestamp_ns: u64) -> BTreeMap<String, f64> {
        self.symbols
            .iter()
            .filter_map(|(symbol, market_id)| {
                self.models
                    .get(market_id)
                    .and_then(|model| model.open_at(timestamp_ns))
                    .map(|price| (symbol.clone(), price))
            })
            .collect()
    }

    fn cancel_candidate(&mut self, candidate_id: &str) {
        for model in self.models.values_mut() {
            model.cancel_candidate(candidate_id);
        }
    }

    fn ordinary_idle_sync_blocked(&self, candidate: &Candidate) -> bool {
        if candidate.policy != "ordinary_idle_eod_open" {
            return false;
        }
        let values = self
            .models
            .values()
            .filter_map(|model| model.last_indicator.as_ref())
            .filter(|indicator| indicator.bar.timestamp_ns == candidate.signal_time_ns)
            .filter_map(|indicator| indicator.ret60_bps)
            .collect::<Vec<_>>();
        if values.len() < 2
            || values
                .iter()
                .filter(|value| **value * candidate.side.sign() > 0.0)
                .count()
                != 3
        {
            return false;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        variance.sqrt() >= 20.0
    }

    fn publish_targets(&mut self) {
        self.targets.clear();
        if self.synchronized_markets.len() != self.models.len() {
            return;
        }
        for candidate in self.arbitrator.active.values() {
            let Some(market_id) = self.symbols.get(&candidate.symbol) else {
                continue;
            };
            let Some(model) = self.models.get(market_id) else {
                continue;
            };
            let (Some(price), Some(updated)) = (model.latest_price, model.latest_at_ns) else {
                continue;
            };
            let contracts =
                candidate.side.sign() * model.config.full_weight_contracts * candidate.weight;
            if !contracts.is_finite() || !price.is_finite() || !candidate.entry_price.is_finite() {
                continue;
            }
            self.targets.insert(
                candidate.symbol.clone(),
                TargetPosition {
                    symbol: candidate.symbol.clone(),
                    target_instrument_id: model.config.target_instrument_id.clone(),
                    exchange_id: model.latest_exchange.clone(),
                    contracts,
                    entry_price: candidate.entry_price,
                    latest_price: price,
                    multiplier: model.config.contract_multiplier,
                    updated_at_ms: (updated / 1_000_000) as i64,
                },
            );
        }
    }

    pub fn targets(&self) -> Vec<TargetPosition> {
        self.targets.values().cloned().collect()
    }

    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    pub fn install_benchmark_target(&mut self) {
        let Some((market_id, model)) = self.models.iter().next() else {
            return;
        };
        self.synchronized_markets.insert(market_id.clone());
        let candidate = Candidate {
            id: "benchmark-target".into(),
            symbol: model.config.symbol.clone(),
            side: Side::Long,
            family: "benchmark",
            policy: "benchmark",
            weight: 1.0,
            signal_time_ns: 0,
            entry_time_ns: 0,
            entry_price: 4_000.0,
            exit_plan: ExitPlan::AtOrAfter(u64::MAX),
            ret30_signed: 0.0,
            ret60_signed: 0.0,
            vwap_signed: 0.0,
            trend_distance_bps: 0.0,
        };
        self.arbitrator
            .active
            .insert(candidate.id.clone(), candidate);
    }

    pub fn last_bar_timestamp_ns(&self, market_instrument_id: &str) -> Option<u64> {
        self.models
            .get(market_instrument_id)
            .and_then(|model| model.last_indicator.as_ref())
            .map(|indicator| indicator.bar.timestamp_ns)
    }

    pub fn clear_all(&mut self) {
        for model in self.models.values_mut() {
            model.clear_live_state();
        }
        self.synchronized_markets.clear();
        self.targets.clear();
    }
}

fn candidate_keys(candidate: &Candidate) -> Vec<String> {
    let side = match candidate.side {
        Side::Long => "long",
        Side::Short => "short",
    };
    vec![
        format!(
            "{}|{}|{}:{}",
            candidate.symbol, side, candidate.family, candidate.policy
        ),
        format!("{}|{}|{}", candidate.symbol, side, candidate.policy),
        format!("{}|{}", candidate.symbol, side),
        format!("{}:{}", candidate.family, candidate.policy),
        candidate.policy.to_owned(),
        "global".into(),
    ]
}

fn idle_key(candidate: &Candidate) -> String {
    let side = match candidate.side {
        Side::Long => "long",
        Side::Short => "short",
    };
    format!(
        "{}|{}|{}|{}|{}",
        candidate.symbol,
        side,
        market_hour(candidate.entry_time_ns).unwrap_or_default(),
        bin(candidate.ret30_signed, 80.0, 120.0, "ret"),
        bin(candidate.vwap_signed, 100.0, 130.0, "vwap")
    )
}

fn bin(value: f64, low: f64, high: f64, name: &str) -> String {
    if value < low {
        format!("{name}_low")
    } else if value <= high {
        format!("{name}_mid")
    } else {
        format!("{name}_high")
    }
}

fn portfolio_weight(candidate: &Candidate) -> f64 {
    if candidate.policy == "ordinary_idle_eod_open"
        || candidate.policy == "trend_recovery_h45_open"
        || candidate.policy == "im_day120_140_h45_open"
        || candidate.policy == "if_vwap_confirm_eod_open"
    {
        candidate.weight
    } else {
        candidate.weight / 4.0
    }
}

fn bad_late_state(row: &Indicator, candidate: &Candidate, bars_to_close: usize) -> bool {
    if bars_to_close == 0 || bars_to_close > 30 {
        return false;
    }
    let (Some(fast), Some(slow), Some(momentum)) =
        (row.fast_ma, row.slow_ma, row.late_momentum_bps)
    else {
        return false;
    };
    let unrealized = candidate
        .side
        .raw_return(candidate.entry_price, row.bar.close)
        * 10_000.0;
    let close_fast = (row.bar.close / fast - 1.0) * 10_000.0 * candidate.side.sign();
    let reversal = -(fast / slow - 1.0) * 10_000.0 * candidate.side.sign();
    let momentum = -momentum * candidate.side.sign();
    unrealized <= -50.0 && close_fast <= -10.0 && reversal >= 10.0 && momentum >= 10.0
}

fn cumulative_delta(current: f64, previous: Option<f64>) -> f64 {
    previous
        .map(|prior| {
            if current >= prior {
                current - prior
            } else {
                current
            }
        })
        .unwrap_or(current)
}

fn trim(values: &mut VecDeque<f64>, maximum: usize) {
    while values.len() > maximum {
        values.pop_front();
    }
}

fn mean_tail(values: &VecDeque<f64>, count: usize) -> Option<f64> {
    (values.len() >= count).then(|| values.iter().rev().take(count).sum::<f64>() / count as f64)
}

fn return_bps(values: &VecDeque<f64>, close: f64, lag: usize) -> Option<f64> {
    (values.len() > lag).then(|| (close / values[values.len() - lag - 1] - 1.0) * 10_000.0)
}

fn z_score(value: f64, values: &[f64], minimum: usize) -> Option<f64> {
    if values.len() < minimum {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|item| (item - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    (variance > 0.0).then(|| (value - mean) / variance.sqrt())
}

fn z_score_from_moments(
    count: usize,
    sum: f64,
    sum_squares: f64,
    value: f64,
    minimum: usize,
) -> Option<f64> {
    if count < minimum {
        return None;
    }
    let mean = sum / count as f64;
    let variance = (sum_squares - sum * sum / count as f64) / (count - 1) as f64;
    (variance > 0.0).then(|| (value - mean) / variance.sqrt())
}

fn quantile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn label_event(candidate: Candidate, exit_price: f64, at_ns: u64, events: &mut Vec<ModelEvent>) {
    events.push(ModelEvent::Label {
        candidate,
        exit_price,
        at_ns,
    });
}

fn market_datetime(timestamp_ns: u64) -> Result<NaiveDateTime, String> {
    Ok(DateTime::<Utc>::from_timestamp_nanos(
        i64::try_from(timestamp_ns).map_err(|_| "timestamp outside chrono range")?,
    )
    .naive_utc())
}

fn market_day(timestamp_ns: u64) -> Result<NaiveDate, String> {
    Ok(market_datetime(timestamp_ns)?.date())
}
fn market_hour(timestamp_ns: u64) -> Result<u32, String> {
    Ok(market_datetime(timestamp_ns)?.hour())
}
fn time_hms(timestamp_ns: u64) -> Result<u32, String> {
    let time = market_datetime(timestamp_ns)?.time();
    Ok(time.hour() * 10_000 + time.minute() * 100 + time.second())
}
fn is_session_end(timestamp_ns: u64) -> Result<bool, String> {
    Ok(time_hms(timestamp_ns)? == 150_000)
}

fn tick_timestamp_ns(tick: &CtpdTick) -> Result<u64, String> {
    let date = NaiveDate::parse_from_str(&tick.action_day, "%Y%m%d")
        .map_err(|_| "invalid CTPD action_day")?;
    let time = NaiveTime::parse_from_str(&tick.update_time, "%H:%M:%S")
        .map_err(|_| "invalid CTPD update_time")?;
    if !(0..1_000).contains(&tick.update_millisec) {
        return Err("invalid CTPD update_millisec".into());
    }
    let datetime = NaiveDateTime::new(date, time)
        + chrono::Duration::milliseconds(i64::from(tick.update_millisec));
    u64::try_from(
        datetime
            .and_utc()
            .timestamp_nanos_opt()
            .ok_or("CTPD timestamp outside range")?,
    )
    .map_err(|_| "CTPD timestamp before epoch".into())
}

fn bar_endpoint_ns(tick_ns: u64) -> Result<u64, String> {
    tick_ns
        .checked_div(NS_PER_MINUTE)
        .and_then(|minute| minute.checked_add(1))
        .and_then(|minute| minute.checked_mul(NS_PER_MINUTE))
        .ok_or_else(|| "CTPD tick timestamp overflow".into())
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, NaiveDateTime};

    use super::{CtpdTick, InstrumentConfig, Portfolio};

    fn config(symbol: &str, market: &str, target: &str) -> InstrumentConfig {
        InstrumentConfig {
            symbol: symbol.into(),
            market_instrument_id: market.into(),
            target_instrument_id: target.into(),
            parquet: "/unused".into(),
            full_weight_contracts: 10.0,
            contract_multiplier: 300.0,
            session_bar_count: 240,
            session_end_time: "15:00:00".into(),
        }
    }

    #[test]
    fn rejects_out_of_order_ticks() {
        let config = config("IF8888", "IDX-CFFEX-IF", "IF2609");
        let mut portfolio = Portfolio::default();
        portfolio.insert(
            config.market_instrument_id.clone(),
            Portfolio::new_model(&config),
        );
        let tick = |minute: i64| {
            let at = NaiveDateTime::parse_from_str("20260810 09:30:00", "%Y%m%d %H:%M:%S").unwrap()
                + Duration::minutes(minute);
            CtpdTick {
                instrument_id: "IDX-CFFEX-IF".into(),
                exchange_id: "CFFEX".into(),
                trading_day: at.format("%Y%m%d").to_string(),
                action_day: at.format("%Y%m%d").to_string(),
                update_time: at.format("%H:%M:%S").to_string(),
                update_millisec: 0,
                last_price: 4_000.0,
                volume: minute as f64 + 1.0,
                turnover: 4_000.0 * (minute + 1) as f64,
                open_interest: 100.0,
            }
        };
        portfolio.ingest(tick(1)).unwrap();
        assert!(
            portfolio
                .ingest(tick(0))
                .unwrap_err()
                .contains("out of order")
        );
    }

    #[test]
    fn target_contract_is_never_the_continuous_market_id() {
        let config = config("IF8888", "IDX-CFFEX-IF", "IF2609");
        assert_ne!(config.market_instrument_id, config.target_instrument_id);
    }
}

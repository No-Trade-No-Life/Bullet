//! Deterministic, configuration-driven bar backtesting for Bullet strategies.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bullet_core::Bar;
use serde::Deserialize;

const NANOS_PER_DAY: u64 = 86_400_000_000_000;
const NANOS_PER_YEAR: f64 = 365.25 * NANOS_PER_DAY as f64;
const TRADING_DAYS_PER_YEAR: f64 = 252.0;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub version: u32,
    pub backtest: BacktestConfig,
    pub execution: ExecutionConfig,
    pub fees: FeesConfig,
    pub instruments: Vec<InstrumentConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BacktestConfig {
    pub mode: Mode,
    pub initial_cash: f64,
    pub currency: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Bar,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExecutionConfig {
    pub fill_price: FillPrice,
    pub slippage_bps: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FillPrice {
    NextBarOpen,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FeesConfig {
    pub mode: FeeMode,
    pub open: f64,
    pub close: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FeeMode {
    PerContract,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InstrumentConfig {
    pub id: String,
    pub data: PathBuf,
    pub multiplier: f64,
    pub margin_rate: f64,
    pub tick_size: f64,
}

impl Config {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, BacktestError> {
        let path = path.as_ref();
        let source = fs::read_to_string(path).map_err(BacktestError::ReadConfig)?;
        let mut config: Self = toml::from_str(&source).map_err(BacktestError::ParseConfig)?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        for instrument in &mut config.instruments {
            instrument.data = expand_home(&instrument.data);
            if instrument.data.is_relative() {
                instrument.data = base.join(&instrument.data);
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), BacktestError> {
        if self.version != 1 {
            return Err(BacktestError::UnsupportedConfigVersion(self.version));
        }
        if !self.backtest.initial_cash.is_finite() || self.backtest.initial_cash <= 0.0 {
            return Err(BacktestError::InvalidConfig(
                "backtest.initial_cash must be positive",
            ));
        }
        if self.instruments.is_empty() {
            return Err(BacktestError::InvalidConfig(
                "at least one instrument is required",
            ));
        }
        if !self.execution.slippage_bps.is_finite() || self.execution.slippage_bps < 0.0 {
            return Err(BacktestError::InvalidConfig(
                "execution.slippage_bps must be non-negative",
            ));
        }
        if !self.fees.open.is_finite()
            || self.fees.open < 0.0
            || !self.fees.close.is_finite()
            || self.fees.close < 0.0
        {
            return Err(BacktestError::InvalidConfig("fees must be non-negative"));
        }
        let mut symbols = BTreeMap::new();
        for instrument in &self.instruments {
            if instrument.id.is_empty() || symbols.insert(&instrument.id, ()).is_some() {
                return Err(BacktestError::InvalidConfig(
                    "instrument ids must be non-empty and unique",
                ));
            }
            if !instrument.multiplier.is_finite()
                || instrument.multiplier <= 0.0
                || !instrument.margin_rate.is_finite()
                || instrument.margin_rate <= 0.0
                || !instrument.tick_size.is_finite()
                || instrument.tick_size <= 0.0
            {
                return Err(BacktestError::InvalidConfig(
                    "instrument multiplier, margin_rate, and tick_size must be positive",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Order {
    None,
    Buy(u64),
    Sell(u64),
    Close,
}

pub trait Strategy {
    fn on_bar(&mut self, context: BarContext<'_>) -> Order;
}

#[derive(Clone, Copy, Debug)]
pub struct BarContext<'a> {
    pub instrument: &'a str,
    pub bar: &'a Bar,
    pub position: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BacktestResult {
    pub bars: usize,
    pub data_size_bytes: u64,
    pub runtime_ms: u128,
    pub fills: usize,
    pub round_trips: usize,
    pub fees_paid: f64,
    pub slippage_paid: f64,
    pub ending_positions: BTreeMap<String, i64>,
    pub performance: Performance,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Performance {
    pub initial_equity: f64,
    pub final_equity: f64,
    pub cumulative_return: f64,
    pub annualized_return: Option<f64>,
    pub annualized_sharpe: Option<f64>,
    pub max_drawdown: f64,
    pub return_drawdown_ratio: Option<f64>,
    pub daily_observations: usize,
}

pub fn run<S: Strategy>(
    config: &Config,
    strategy: &mut S,
) -> Result<BacktestResult, BacktestError> {
    let started = Instant::now();
    let mut events = Vec::new();
    let mut data_size_bytes = 0;
    for instrument in &config.instruments {
        data_size_bytes += fs::metadata(&instrument.data)
            .map_err(BacktestError::ReadData)?
            .len();
        for bar in bullet_data::read_bars(&instrument.data).map_err(BacktestError::Data)? {
            events.push(EventBar {
                instrument: instrument.id.clone(),
                bar,
            });
        }
    }
    events.sort_by(|left, right| {
        left.bar
            .timestamp_ns
            .cmp(&right.bar.timestamp_ns)
            .then_with(|| left.instrument.cmp(&right.instrument))
    });
    if events.is_empty() {
        return Err(BacktestError::EmptyBars);
    }

    let instruments: BTreeMap<_, _> = config
        .instruments
        .iter()
        .map(|value| (value.id.as_str(), value))
        .collect();
    let mut state = State::new(config.backtest.initial_cash, &config.instruments);
    let mut daily_equity = Vec::new();

    for event in &events {
        let instrument = instruments[&event.instrument.as_str()];
        state.fill_pending(
            &event.instrument,
            event.bar.open.value(),
            instrument,
            &instruments,
            config,
        )?;
        state
            .latest_closes
            .insert(event.instrument.clone(), event.bar.close.value());
        let order = strategy.on_bar(BarContext {
            instrument: &event.instrument,
            bar: &event.bar,
            position: state.positions[&event.instrument],
        });
        state.submit(&event.instrument, order)?;
        record_daily_equity(
            &mut daily_equity,
            event.bar.timestamp_ns,
            state.equity(&instruments),
        );
    }

    let first = events.first().expect("non-empty events").bar.timestamp_ns;
    let last = events.last().expect("non-empty events").bar.timestamp_ns;
    Ok(BacktestResult {
        bars: events.len(),
        data_size_bytes,
        runtime_ms: started.elapsed().as_millis(),
        fills: state.fills,
        round_trips: state.round_trips,
        fees_paid: state.fees_paid,
        slippage_paid: state.slippage_paid,
        ending_positions: state.positions,
        performance: evaluate(&daily_equity, config.backtest.initial_cash, first, last),
    })
}

struct EventBar {
    instrument: String,
    bar: Bar,
}
struct PendingOrder {
    side: Side,
    quantity: u64,
}
#[derive(Clone, Copy)]
enum Side {
    Buy,
    Sell,
}
struct State {
    cash: f64,
    positions: BTreeMap<String, i64>,
    pending: BTreeMap<String, PendingOrder>,
    latest_closes: BTreeMap<String, f64>,
    fills: usize,
    round_trips: usize,
    fees_paid: f64,
    slippage_paid: f64,
}
impl State {
    fn new(cash: f64, instruments: &[InstrumentConfig]) -> Self {
        Self {
            cash,
            positions: instruments.iter().map(|x| (x.id.clone(), 0)).collect(),
            pending: BTreeMap::new(),
            latest_closes: BTreeMap::new(),
            fills: 0,
            round_trips: 0,
            fees_paid: 0.0,
            slippage_paid: 0.0,
        }
    }
    fn submit(&mut self, instrument: &str, order: Order) -> Result<(), BacktestError> {
        let position = self.positions[instrument];
        let pending = match order {
            Order::None => return Ok(()),
            Order::Buy(quantity) => PendingOrder {
                side: Side::Buy,
                quantity,
            },
            Order::Sell(quantity) => PendingOrder {
                side: Side::Sell,
                quantity,
            },
            Order::Close => PendingOrder {
                side: Side::Sell,
                quantity: position.unsigned_abs(),
            },
        };
        if pending.quantity == 0 {
            return Ok(());
        }
        if self
            .pending
            .insert(instrument.to_owned(), pending)
            .is_some()
        {
            return Err(BacktestError::PendingOrder(instrument.to_owned()));
        }
        Ok(())
    }
    fn fill_pending(
        &mut self,
        symbol: &str,
        open: f64,
        instrument: &InstrumentConfig,
        instruments: &BTreeMap<&str, &InstrumentConfig>,
        config: &Config,
    ) -> Result<(), BacktestError> {
        let Some(order) = self.pending.remove(symbol) else {
            return Ok(());
        };
        let position = self.positions[symbol];
        if matches!(order.side, Side::Sell) && order.quantity as i64 > position {
            return Err(BacktestError::InsufficientPosition {
                instrument: symbol.to_owned(),
                position,
                requested: order.quantity,
            });
        }
        let adjustment = open * config.execution.slippage_bps / 10_000.0;
        let price = match order.side {
            Side::Buy => open + adjustment,
            Side::Sell => open - adjustment,
        };
        let notional = price * order.quantity as f64 * instrument.multiplier;
        if matches!(order.side, Side::Buy) {
            let required_margin = (position + order.quantity as i64) as f64 * notional
                / order.quantity as f64
                * instrument.margin_rate;
            let equity = self.equity(instruments);
            if required_margin > equity {
                return Err(BacktestError::InsufficientMargin {
                    instrument: symbol.to_owned(),
                    required: required_margin,
                    equity,
                });
            }
        }
        let fee = match order.side {
            Side::Buy => config.fees.open,
            Side::Sell => config.fees.close,
        } * order.quantity as f64;
        match order.side {
            Side::Buy => {
                self.cash -= notional + fee;
                self.positions
                    .insert(symbol.to_owned(), position + order.quantity as i64);
            }
            Side::Sell => {
                self.cash += notional - fee;
                self.positions
                    .insert(symbol.to_owned(), position - order.quantity as i64);
                self.round_trips += 1;
            }
        }
        self.fills += 1;
        self.fees_paid += fee;
        self.slippage_paid += adjustment * order.quantity as f64 * instrument.multiplier;
        Ok(())
    }
    fn equity(&self, instruments: &BTreeMap<&str, &InstrumentConfig>) -> f64 {
        self.positions
            .iter()
            .fold(self.cash, |equity, (symbol, position)| {
                equity
                    + self.latest_closes.get(symbol).map_or(0.0, |price| {
                        *price * *position as f64 * instruments[symbol.as_str()].multiplier
                    })
            })
    }
}

fn expand_home(path: &Path) -> PathBuf {
    path.strip_prefix("~")
        .ok()
        .and_then(|suffix| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(suffix)))
        .unwrap_or_else(|| path.to_owned())
}

fn record_daily_equity(values: &mut Vec<(u64, f64)>, timestamp_ns: u64, equity: f64) {
    let day = timestamp_ns / NANOS_PER_DAY;
    if let Some((current_day, current_equity)) = values.last_mut()
        && *current_day == day
    {
        *current_equity = equity;
    } else {
        values.push((day, equity));
    }
}
fn evaluate(values: &[(u64, f64)], initial: f64, first: u64, last: u64) -> Performance {
    let final_equity = values.last().expect("events create equity").1;
    let cumulative_return = final_equity / initial - 1.0;
    let annualized_return = (last > first)
        .then(|| (final_equity / initial).powf(NANOS_PER_YEAR / (last - first) as f64) - 1.0);
    let max_drawdown = values
        .iter()
        .fold((initial, 0.0_f64), |(peak, drawdown), (_, equity)| {
            let peak = peak.max(*equity);
            (peak, drawdown.min(*equity / peak - 1.0))
        })
        .1;
    let returns: Vec<_> = values
        .iter()
        .scan(initial, |previous, (_, equity)| {
            let value = *equity / *previous - 1.0;
            *previous = *equity;
            Some(value)
        })
        .collect();
    let annualized_sharpe = (returns.len() > 1)
        .then(|| {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            (variance > 0.0).then(|| mean / variance.sqrt() * TRADING_DAYS_PER_YEAR.sqrt())
        })
        .flatten();
    Performance {
        initial_equity: initial,
        final_equity,
        cumulative_return,
        annualized_return,
        annualized_sharpe,
        max_drawdown,
        return_drawdown_ratio: annualized_return
            .filter(|_| max_drawdown < 0.0)
            .map(|value| value / -max_drawdown),
        daily_observations: values.len(),
    }
}

#[derive(Debug)]
pub enum BacktestError {
    ReadConfig(std::io::Error),
    ParseConfig(toml::de::Error),
    UnsupportedConfigVersion(u32),
    InvalidConfig(&'static str),
    ReadData(std::io::Error),
    Data(bullet_data::DataError),
    EmptyBars,
    PendingOrder(String),
    InsufficientPosition {
        instrument: String,
        position: i64,
        requested: u64,
    },
    InsufficientMargin {
        instrument: String,
        required: f64,
        equity: f64,
    },
}
impl fmt::Display for BacktestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadConfig(e) => write!(f, "cannot read config: {e}"),
            Self::ParseConfig(e) => write!(f, "cannot parse config: {e}"),
            Self::UnsupportedConfigVersion(v) => write!(f, "unsupported config version {v}"),
            Self::InvalidConfig(message) => f.write_str(message),
            Self::ReadData(e) => write!(f, "cannot read data metadata: {e}"),
            Self::Data(e) => write!(f, "cannot read bars: {e}"),
            Self::EmptyBars => f.write_str("backtest requires at least one bar"),
            Self::PendingOrder(symbol) => write!(f, "pending order already exists for {symbol}"),
            Self::InsufficientPosition {
                instrument,
                position,
                requested,
            } => write!(
                f,
                "cannot sell {requested} {instrument}; position is {position}"
            ),
            Self::InsufficientMargin {
                instrument,
                required,
                equity,
            } => write!(
                f,
                "cannot open {instrument}; required margin {required:.6} exceeds equity {equity:.6}"
            ),
        }
    }
}
impl Error for BacktestError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ReadConfig(e) | Self::ReadData(e) => Some(e),
            Self::ParseConfig(e) => Some(e),
            Self::Data(e) => Some(e),
            _ => None,
        }
    }
}

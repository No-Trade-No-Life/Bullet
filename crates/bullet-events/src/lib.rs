use bullet_core::MarketEvent;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EventLogError {
    Sequence { expected: u64, actual: u64 },
    Timestamp { previous: i128, actual: i128 },
}

#[derive(Default)]
pub struct CausalEventLog {
    events: Vec<MarketEvent>,
}

impl CausalEventLog {
    pub fn append(&mut self, event: MarketEvent) -> Result<(), EventLogError> {
        let expected = self.events.len() as u64;
        if event.sequence != expected {
            return Err(EventLogError::Sequence {
                expected,
                actual: event.sequence,
            });
        }
        if let Some(previous) = self.events.last()
            && event.timestamp_ns < previous.timestamp_ns
        {
            return Err(EventLogError::Timestamp {
                previous: previous.timestamp_ns,
                actual: event.timestamp_ns,
            });
        }
        self.events.push(event);
        Ok(())
    }

    pub fn events(&self) -> &[MarketEvent] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use bullet_core::{EventKind, MarketEvent};

    use super::{CausalEventLog, EventLogError};

    fn event(sequence: u64, timestamp_ns: i128) -> MarketEvent {
        MarketEvent {
            sequence,
            timestamp_ns,
            instrument: "ES".to_owned(),
            kind: EventKind::BarOpen { price: 5_000 },
        }
    }

    #[test]
    fn accepts_one_causal_sequence() {
        let mut log = CausalEventLog::default();
        assert_eq!(log.append(event(0, 10)), Ok(()));
        assert_eq!(log.append(event(1, 11)), Ok(()));
        assert_eq!(log.events().len(), 2);
    }

    #[test]
    fn rejects_non_causal_sequence() {
        let mut log = CausalEventLog::default();
        assert_eq!(log.append(event(0, 10)), Ok(()));
        assert_eq!(
            log.append(event(2, 11)),
            Err(EventLogError::Sequence {
                expected: 1,
                actual: 2,
            })
        );
    }
}

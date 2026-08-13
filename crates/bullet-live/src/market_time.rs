use chrono::{DateTime, FixedOffset, NaiveDateTime, TimeZone, Utc};

pub(crate) fn shanghai() -> FixedOffset {
    FixedOffset::east_opt(8 * 3_600).expect("China Standard Time offset is valid")
}

pub(crate) fn market_datetime(timestamp_ns: u64) -> Result<NaiveDateTime, String> {
    Ok(DateTime::<Utc>::from_timestamp_nanos(
        i64::try_from(timestamp_ns).map_err(|_| "timestamp outside chrono range")?,
    )
    .with_timezone(&shanghai())
    .naive_local())
}

pub(crate) fn timestamp_ns_from_shanghai_wall_clock(
    datetime: NaiveDateTime,
) -> Result<u64, String> {
    let instant = shanghai()
        .from_local_datetime(&datetime)
        .single()
        .ok_or("Shanghai market timestamp is not unambiguous")?;
    u64::try_from(
        instant
            .timestamp_nanos_opt()
            .ok_or("market timestamp outside chrono range")?,
    )
    .map_err(|_| "market timestamp before epoch".into())
}

pub(crate) fn timestamp_ns_from_ctpd_ms(timestamp_ms: i64) -> Result<u64, String> {
    let nanos = timestamp_ms
        .checked_mul(1_000_000)
        .ok_or("CTPD Kline timestamp overflow")?;
    u64::try_from(nanos).map_err(|_| "CTPD Kline timestamp before epoch".into())
}

pub(crate) fn ctpd_ms_from_timestamp_ns(timestamp_ns: u64) -> Result<i64, String> {
    let timestamp = i64::try_from(timestamp_ns).map_err(|_| "timestamp outside chrono range")?;
    Ok(DateTime::<Utc>::from_timestamp_nanos(timestamp).timestamp_millis())
}

#[cfg(test)]
mod tests {
    use chrono::{NaiveDateTime, TimeZone, Utc};

    use super::{market_datetime, timestamp_ns_from_shanghai_wall_clock};

    #[test]
    fn translates_shanghai_wall_clock_to_a_real_utc_instant() {
        let market = NaiveDateTime::parse_from_str("20260812 11:10:00", "%Y%m%d %H:%M:%S").unwrap();
        let timestamp_ns = timestamp_ns_from_shanghai_wall_clock(market).unwrap();
        let utc = Utc
            .timestamp_nanos(i64::try_from(timestamp_ns).unwrap())
            .to_rfc3339();

        assert_eq!(utc, "2026-08-12T03:10:00+00:00");
        assert_eq!(market_datetime(timestamp_ns).unwrap(), market);
    }
}

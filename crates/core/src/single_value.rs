//! Single-value interval detection compatible with `qlib.utils.time.is_single_value`.

use chrono::{NaiveTime, TimeDelta, Timelike};

use crate::Region;

/// Decide whether a market interval can contain only one stored value.
///
/// `elapsed` should be computed as `end - start` by the caller so timezone-aware
/// adapters can preserve absolute-time subtraction while supplying `start_time` in
/// the market's local clock. The comparison is intentionally strict.
#[must_use]
pub fn is_single_market_value(
    start_time: NaiveTime,
    elapsed: TimeDelta,
    frequency: TimeDelta,
    region: Region,
) -> bool {
    if elapsed < frequency {
        return true;
    }
    let whole_second = start_time.second() == 0;
    match region {
        Region::Cn => {
            whole_second
                && ((start_time.hour() == 11 && start_time.minute() == 29)
                    || (start_time.hour() == 14 && start_time.minute() == 59))
        }
        Region::Tw => whole_second && start_time.hour() == 13 && start_time.minute() >= 25,
        Region::Us => whole_second && start_time.hour() == 15 && start_time.minute() == 59,
    }
}

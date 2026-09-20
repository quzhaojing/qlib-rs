//! Python builtin temporal identity produced by `NumPy` object boxing.
use super::{BuiltinFrameValue as B, TemporalFrameValue as T, TupleFrameValue as V};
use arrow_schema::{ArrowError, TimeUnit};

const DAY_US: i128 = 86_400_000_000;
const DATE_MIN: i128 = -62_135_596_800_000_000;
const DATE_MAX: i128 = 253_402_300_799_999_999;
const DELTA_MIN: i128 = -999_999_999 * DAY_US;
const DELTA_MAX: i128 = 1_000_000_000 * DAY_US - 1;

/// Python builtin scalars, distinct from Pandas Timestamp/Timedelta objects.
/// Datetimes are naive microseconds from the Unix epoch (years 1..=9999).
/// Timedeltas retain the full Python range, which exceeds i64 microseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PythonTemporalValue {
    Datetime { microseconds: i64 },
    Timedelta { microseconds: i128 },
}

fn invalid() -> ArrowError {
    ArrowError::InvalidArgumentError("invalid Python temporal object payload".into())
}

impl PythonTemporalValue {
    pub(super) fn validate(self) -> Result<(), ArrowError> {
        let valid = match self {
            Self::Datetime { microseconds } => {
                (DATE_MIN..=DATE_MAX).contains(&i128::from(microseconds))
            }
            Self::Timedelta { microseconds } => (DELTA_MIN..=DELTA_MAX).contains(&microseconds),
        };
        if valid { Ok(()) } else { Err(invalid()) }
    }

    pub(super) fn encode(self) -> Result<(u8, T), ArrowError> {
        self.validate()?;
        Ok(match self {
            Self::Datetime { microseconds } => (3, T::Builtin(B::Int(microseconds))),
            Self::Timedelta { microseconds } => (
                4,
                T::Builtin(B::Text(crate::RlCheckpointText::from_utf8(
                    &microseconds.to_string(),
                ))),
            ),
        })
    }

    pub(super) fn decode(tag: u8, payload: T) -> Result<Self, ArrowError> {
        let result = match (tag, payload) {
            (3, T::Builtin(B::Int(microseconds))) => Self::Datetime { microseconds },
            (4, T::Builtin(B::Text(text))) => {
                let text = text.to_utf8().map_err(|_| invalid())?;
                let microseconds = text.parse::<i128>().map_err(|_| invalid())?;
                if text != microseconds.to_string() {
                    return Err(invalid());
                }
                Self::Timedelta { microseconds }
            }
            _ => return Err(invalid()),
        };
        result.validate()?;
        Ok(result)
    }

    pub(super) fn key(self) -> Result<super::object_factorization::Key, ArrowError> {
        self.validate()?;
        Ok(match self {
            Self::Datetime { microseconds } => {
                super::object_factorization::Key::Timestamp(i128::from(microseconds) * 1000, false)
            }
            Self::Timedelta { microseconds } => {
                super::object_factorization::Key::Duration(microseconds * 1000)
            }
        })
    }

    pub(super) fn inference_value(self) -> Option<T> {
        match self {
            Self::Datetime { microseconds } => Some(T::Timestamp {
                ticks: microseconds,
                unit: TimeUnit::Microsecond,
                timezone: None,
            }),
            Self::Timedelta { microseconds } => {
                i64::try_from(microseconds).ok().map(|ticks| T::Duration {
                    ticks,
                    unit: TimeUnit::Microsecond,
                })
            }
        }
    }
}

pub(super) fn numpy_box(ticks: i64, unit: TimeUnit, datetime: bool) -> V {
    let scale = match unit {
        TimeUnit::Second => 1_000_000,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1,
        TimeUnit::Nanosecond => return V::Scalar(T::Builtin(B::Int(ticks))),
    };
    let microseconds = i128::from(ticks) * scale;
    let value = if datetime {
        i64::try_from(microseconds)
            .ok()
            .map(|microseconds| PythonTemporalValue::Datetime { microseconds })
    } else {
        Some(PythonTemporalValue::Timedelta { microseconds })
    };
    if let Some(value) = value.filter(|v| v.validate().is_ok()) {
        V::PythonTemporal(value)
    } else {
        V::Scalar(T::Builtin(B::Int(ticks)))
    }
}

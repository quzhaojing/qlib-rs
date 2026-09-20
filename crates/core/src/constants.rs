//! Runtime constants compatible with `qlib.constant`.

use chrono::TimeDelta;
use ndarray::{ArrayBase, Dimension, RawData};
use serde::{Deserialize, Serialize};
use strum::{AsRefStr, Display, EnumString, VariantArray};

/// Mainland China market region identifier.
pub const REG_CN: &str = "cn";

/// United States market region identifier.
pub const REG_US: &str = "us";

/// Taiwan market region identifier.
pub const REG_TW: &str = "tw";

/// Epsilon used by Qlib to avoid division by zero.
pub const EPS: f64 = 1e-12;

/// A finite integer sentinel used as practical infinity by Qlib.
pub const INF: i64 = 1_000_000_000_000_000_000;

/// One calendar day, matching `pandas.Timedelta("1day")`.
pub const ONE_DAY: TimeDelta = TimeDelta::days(1);

/// One minute, matching `pandas.Timedelta("1min")`.
pub const ONE_MIN: TimeDelta = TimeDelta::minutes(1);

/// One second used to exclude the right endpoint of a time interval.
pub const EPS_T: TimeDelta = TimeDelta::seconds(1);

/// Static counterpart of Python's `float_or_ndarray` type variable.
///
/// The upstream marker constrains values to either the built-in `float` or an
/// `np.ndarray`, without constraining the array's element type or dimension.
pub trait FloatOrNdarray {}

impl FloatOrNdarray for f64 {}

impl<S, D> FloatOrNdarray for ArrayBase<S, D>
where
    S: RawData,
    D: Dimension,
{
}

/// Markets with region-specific defaults in Qlib configuration.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    Display,
    EnumString,
    AsRefStr,
    VariantArray,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Region {
    /// Mainland China.
    Cn,
    /// United States.
    Us,
    /// Taiwan.
    Tw,
}

impl Region {
    /// Return the exact lowercase code used by the Python configuration API.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Cn => REG_CN,
            Self::Us => REG_US,
            Self::Tw => REG_TW,
        }
    }
}

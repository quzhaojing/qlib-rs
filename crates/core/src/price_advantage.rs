//! Real numeric price advantages from `qlib.rl.order_execution.utils`.

use half::f16;
use ndarray::{Array, ArrayView, Dimension};
use num_traits::AsPrimitive;
use thiserror::Error;

/// Invalid integer direction, with the upstream diagnostic spelling.
#[derive(Debug, PartialEq, Eq, Error)]
#[error("Unexpected order direction: {0}")]
pub struct PriceAdvantageError(pub i64);

/// The upstream zero-baseline path preserves input dtype, while division may
/// promote integer input. A nonzero-baseline singleton becomes a Python float.
#[derive(Debug, PartialEq)]
pub enum ArrayPriceAdvantage<T: PriceElement, D: Dimension> {
    ZeroBaseline(Array<T, D>),
    Scalar(f64),
    Array(Array<T::Float, D>),
}

/// Real `NumPy` arithmetic precision used by a supported input dtype.
pub trait PriceFloat: Copy {
    #[must_use]
    fn calculate(self, baseline: f64, buy: bool) -> Self;
    fn as_f64(self) -> f64;
}

/// Real numeric ndarray dtype and its true-division promotion.
pub trait PriceElement: Copy + Default {
    type Float: PriceFloat;
    fn promoted(self) -> Self::Float;
}

macro_rules! float_type {
    ($ty:ty, $convert:expr, $widen:expr) => {
        impl PriceFloat for $ty {
            fn calculate(self, baseline: f64, buy: bool) -> Self {
                let one = ($convert)(1.0);
                let ratio = self / ($convert)(baseline);
                let difference = if buy { one - ratio } else { ratio - one };
                let value = difference * ($convert)(10_000.0);
                if value.is_nan() {
                    Self::default()
                } else if value == Self::INFINITY {
                    Self::MAX
                } else if value == Self::NEG_INFINITY {
                    Self::MIN
                } else {
                    value
                }
            }

            fn as_f64(self) -> f64 {
                ($widen)(self)
            }
        }

        impl PriceElement for $ty {
            type Float = Self;
            fn promoted(self) -> Self {
                self
            }
        }
    };
}

float_type!(f64, std::convert::identity, std::convert::identity);
float_type!(f32, narrow_f32, f64::from);
float_type!(f16, f16::from_f64, f16::to_f64);

// NumPy casts a Python scalar to the array's arithmetic precision before each
// ufunc, including rounding and overflow. This truncation is intentional.
#[allow(clippy::cast_possible_truncation)]
fn narrow_f32(value: f64) -> f32 {
    value as f32
}

macro_rules! integral_type {
    ($($ty:ty),+) => {$(
        impl PriceElement for $ty {
            type Float = f64;
            // True division promotes all NumPy integer widths to f64, losing
            // low bits for large i64/u64 exactly as this conversion does.
            fn promoted(self) -> f64 {
                self.as_()
            }
        }
    )+};
}

integral_type!(i8, u8, i16, u16, i32, u32, i64, u64);

impl PriceElement for bool {
    type Float = f64;
    fn promoted(self) -> f64 {
        f64::from(u8::from(self))
    }
}

pub(crate) fn buy_direction(direction: i64) -> Result<bool, PriceAdvantageError> {
    match direction {
        1 => Ok(true),
        0 => Ok(false),
        other => Err(PriceAdvantageError(other)),
    }
}

/// Calculate the Python-float input path, including its early zero-baseline return.
///
/// # Errors
/// Returns the source `ValueError` diagnostic for an integer other than 0 or 1,
/// unless the baseline is zero.
pub fn price_advantage(
    execution_price: f64,
    baseline_price: f64,
    direction: i64,
) -> Result<f64, PriceAdvantageError> {
    if baseline_price == 0.0 {
        return Ok(0.0);
    }
    Ok(execution_price.calculate(baseline_price, buy_direction(direction)?))
}

/// Calculate real numeric arrays without changing input values or shape.
///
/// Supports bool, all fixed-width integer types, f16, f32 and f64. For complex
/// arrays see [`crate::complex_price_advantage::complex_price_advantage`]. Object,
/// datetime and extended-precision dtypes require separate adapters.
/// Views, including non-contiguous and reversed views, are read in logical order.
///
/// # Errors
/// Rejects invalid directions after the zero-baseline short circuit, even for
/// an empty array.
pub fn price_advantage_array<T: PriceElement, D: Dimension>(
    prices: &ArrayView<'_, T, D>,
    baseline_price: f64,
    direction: i64,
) -> Result<ArrayPriceAdvantage<T, D>, PriceAdvantageError> {
    if baseline_price == 0.0 {
        return Ok(ArrayPriceAdvantage::ZeroBaseline(
            prices.mapv(|_| T::default()),
        ));
    }
    let buy = buy_direction(direction)?;
    let values = prices.mapv(|value| value.promoted().calculate(baseline_price, buy));
    if let Some(value) = values.iter().next().filter(|_| values.len() == 1) {
        Ok(ArrayPriceAdvantage::Scalar(value.as_f64()))
    } else {
        Ok(ArrayPriceAdvantage::Array(values))
    }
}

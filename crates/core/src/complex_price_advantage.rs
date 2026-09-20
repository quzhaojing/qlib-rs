//! Complex ndarray price advantages with NumPy-compatible operation order.

use ndarray::{Array, ArrayView, Dimension};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};

use crate::price_advantage::{PriceAdvantageError, buy_direction};

/// Component precision for complex64 and complex128 arrays.
pub trait ComplexPriceComponent: Float + AsPrimitive<f64> {
    fn from_baseline(value: f64) -> Self;
}

impl ComplexPriceComponent for f32 {
    fn from_baseline(value: f64) -> Self {
        value.as_()
    }
}

impl ComplexPriceComponent for f64 {
    fn from_baseline(value: f64) -> Self {
        value
    }
}

/// A singleton becomes a Python-compatible double-precision complex scalar.
#[derive(Debug, PartialEq)]
pub enum ComplexPriceAdvantage<T, D: Dimension> {
    Scalar(Complex<f64>),
    Array(Array<Complex<T>, D>),
}

fn finite<T: Float>(value: T) -> T {
    if value.is_nan() {
        T::zero()
    } else if value == T::infinity() {
        T::max_value()
    } else if value == T::neg_infinity() {
        -T::max_value()
    } else {
        value
    }
}

fn calculate<T: ComplexPriceComponent>(price: Complex<T>, baseline: T, buy: bool) -> Complex<T> {
    let zero = T::zero();
    let one = T::one();
    // NumPy's complex divide loop uses reciprocal scaling and cross terms,
    // even with a real denominator. num-complex's scalar divide omits these
    // terms, changing rounding and infinity/NaN propagation. A narrowed f32
    // baseline can be zero although the original f64 baseline was nonzero.
    let ratio = if baseline == zero {
        Complex::new(price.re / zero, price.im / zero)
    } else {
        let cross = zero / baseline;
        let scale = one / (baseline + zero * cross);
        Complex::new(
            (price.re + price.im * cross) * scale,
            (price.im - price.re * cross) * scale,
        )
    };
    let unit = Complex::new(one, zero);
    let difference = if buy { unit - ratio } else { ratio - unit };
    // Keep the complex multiplication: infinity times the zero imaginary
    // component is observable before nan_to_num cleans each component.
    let result = difference * Complex::new(T::from_baseline(10_000.0), zero);
    Complex::new(finite(result.re), finite(result.im))
}

/// Calculate complex64/complex128 arrays while retaining shape and dtype.
///
/// A zero original baseline returns a fresh zero array, including singleton
/// and zero-dimensional inputs. Otherwise a singleton is a complex scalar.
/// Numeric results follow `NumPy`'s default warning policy; Python warning
/// callbacks and floating-point exception configuration remain an edge concern.
///
/// # Errors
/// Invalid integer directions fail after the zero-baseline short circuit.
pub fn complex_price_advantage<T: ComplexPriceComponent, D: Dimension>(
    prices: &ArrayView<'_, Complex<T>, D>,
    baseline: f64,
    direction: i64,
) -> Result<ComplexPriceAdvantage<T, D>, PriceAdvantageError> {
    if baseline == 0.0 {
        return Ok(ComplexPriceAdvantage::Array(
            prices.mapv(|_| Complex::new(T::zero(), T::zero())),
        ));
    }
    let buy = buy_direction(direction)?;
    let baseline = T::from_baseline(baseline);
    let values = prices.mapv(|price| calculate(price, baseline, buy));
    if let Some(value) = values.iter().next().filter(|_| values.len() == 1) {
        Ok(ComplexPriceAdvantage::Scalar(Complex::new(
            value.re.as_(),
            value.im.as_(),
        )))
    } else {
        Ok(ComplexPriceAdvantage::Array(values))
    }
}

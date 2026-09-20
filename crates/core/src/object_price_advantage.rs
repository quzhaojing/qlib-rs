//! Object-array arithmetic stages for `qlib.rl.order_execution.utils`.

use std::convert::Infallible;

use ndarray::{ArrayD, ArrayViewD};
use thiserror::Error;

use crate::price_advantage::{PriceAdvantageError, buy_direction};

/// Object operations may return different types at each arithmetic stage.
pub trait ObjectPriceArithmetic {
    type Input;
    type Quotient;
    type Difference;
    type Value;
    type Error;

    /// # Errors
    /// Propagates the object's division failure.
    fn divide(&mut self, value: &Self::Input, baseline: f64)
    -> Result<Self::Quotient, Self::Error>;
    /// Calculate `1 - value` for buys and `value - 1` for sells.
    /// # Errors
    /// Propagates the object's forward/reverse subtraction failure.
    fn subtract(
        &mut self,
        value: &Self::Quotient,
        buy: bool,
    ) -> Result<Self::Difference, Self::Error>;
    /// Multiply by the Python integer 10000, retaining the returned object.
    /// # Errors
    /// Propagates the object's multiplication failure.
    fn multiply(&mut self, value: &Self::Difference) -> Result<Self::Value, Self::Error>;
    /// Handle the zero-dimensional scalar's `nan_to_num`, `.size` and `.item()`.
    /// Nonzero-dimensional object arrays bypass this hook even when singleton.
    /// # Errors
    /// Propagates scalar conversion or missing-property failures.
    fn finish_scalar(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
}

/// `zeros_like(object)` produces integer-zero elements, independent of input type.
#[derive(Debug, PartialEq)]
pub enum ObjectPriceAdvantage<Value> {
    ZeroBaseline(ArrayD<i64>),
    Scalar(Value),
    Array(ArrayD<Value>),
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum ObjectPriceAdvantageError<Failure> {
    #[error(transparent)]
    Direction(#[from] PriceAdvantageError),
    #[error("{0}")]
    Operation(Failure),
}

fn keep_order_axes<T>(prices: &ArrayViewD<'_, T>) -> Vec<usize> {
    // NumPy starts with the fastest C axis. Zero/broadcast strides are
    // ambiguous and must not displace another axis merely because they are 0.
    let strides: Vec<_> = prices
        .shape()
        .iter()
        .zip(prices.strides())
        .map(
            |(&size, &stride)| {
                if size == 1 { 0 } else { stride.unsigned_abs() }
            },
        )
        .collect();
    let mut axes: Vec<_> = (0..prices.ndim()).rev().collect();
    for current in 1..axes.len() {
        let selected = axes[current];
        let mut destination = current;
        for earlier in (0..current).rev() {
            let previous = axes[earlier];
            if strides[selected] != 0 && strides[previous] != 0 {
                if strides[previous] <= strides[selected] {
                    break;
                }
                destination = earlier;
            }
        }
        axes[destination..=current].rotate_right(1);
    }
    axes.reverse();
    axes
}

/// Run complete object ufunc stages in keep-order axis traversal.
///
/// The input is borrowed, values are never cloned, and earlier object effects
/// remain observable when a later operation fails. Negative-stride axes retain
/// their logical direction; positive axis permutation follows absolute strides.
/// No numeric cleanup is applied to nonzero-dimensional object arrays.
///
/// # Errors
/// Invalid direction or the first arithmetic/scalar-finalization failure.
///
/// # Panics
/// Asserts internal element-count and shape invariants after successful stages.
/// These invariants hold for every valid ndarray view and completed callback.
pub fn object_price_advantage<A: ObjectPriceArithmetic>(
    prices: &ArrayViewD<'_, A::Input>,
    baseline: f64,
    direction: i64,
    arithmetic: &mut A,
) -> Result<ObjectPriceAdvantage<A::Value>, ObjectPriceAdvantageError<A::Error>> {
    if baseline == 0.0 {
        return Ok(ObjectPriceAdvantage::ZeroBaseline(ArrayD::zeros(
            prices.raw_dim(),
        )));
    }
    let buy = buy_direction(direction)?;
    let axes = keep_order_axes(prices);
    let ordered = prices.view().permuted_axes(axes.clone());
    let quotients = ordered
        .iter()
        .map(|value| arithmetic.divide(value, baseline))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ObjectPriceAdvantageError::Operation)?;
    let differences = quotients
        .iter()
        .map(|value| arithmetic.subtract(value, buy))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ObjectPriceAdvantageError::Operation)?;
    drop(quotients);
    let mut values = differences
        .iter()
        .map(|value| arithmetic.multiply(value))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ObjectPriceAdvantageError::Operation)?;
    drop(differences);
    if values.len() == 1 {
        let value = values.pop().expect("one element was checked");
        let value = if prices.ndim() == 0 {
            arithmetic
                .finish_scalar(value)
                .map_err(ObjectPriceAdvantageError::Operation)?
        } else {
            value
        };
        return Ok(ObjectPriceAdvantage::Scalar(value));
    }
    let mut inverse = vec![0; axes.len()];
    for (position, &axis) in axes.iter().enumerate() {
        inverse[axis] = position;
    }
    let result = ArrayD::from_shape_vec(ordered.raw_dim(), values)
        .expect("every stage preserves the number of input elements");
    Ok(ObjectPriceAdvantage::Array(result.permuted_axes(inverse)))
}

/// Concrete adapter for object arrays containing Python-float values.
pub struct FloatObjectArithmetic;

impl ObjectPriceArithmetic for FloatObjectArithmetic {
    type Input = f64;
    type Quotient = f64;
    type Difference = f64;
    type Value = f64;
    type Error = Infallible;

    fn divide(&mut self, value: &f64, baseline: f64) -> Result<f64, Infallible> {
        Ok(value / baseline)
    }

    fn subtract(&mut self, value: &f64, buy: bool) -> Result<f64, Infallible> {
        Ok(if buy { 1.0 - value } else { value - 1.0 })
    }

    fn multiply(&mut self, value: &f64) -> Result<f64, Infallible> {
        Ok(value * 10_000.0)
    }

    fn finish_scalar(&mut self, value: f64) -> Result<f64, Infallible> {
        Ok(if value.is_nan() {
            0.0
        } else {
            value.clamp(f64::MIN, f64::MAX)
        })
    }
}

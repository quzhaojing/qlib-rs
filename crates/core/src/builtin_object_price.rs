//! Python built-in payloads for object-array price arithmetic.

use num_bigint::BigInt;
use num_complex::Complex;
use num_traits::ToPrimitive;
use strum::Display;
use thiserror::Error;

use crate::object_price_advantage::ObjectPriceArithmetic;

/// Non-numeric built-ins fail at division without examining their payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Display)]
pub enum BuiltinPriceType {
    #[strum(serialize = "NoneType")]
    None,
    #[strum(serialize = "str")]
    String,
    #[strum(serialize = "bytes")]
    Bytes,
    #[strum(serialize = "list")]
    List,
    #[strum(serialize = "tuple")]
    Tuple,
    #[strum(serialize = "dict")]
    Dict,
    #[strum(serialize = "set")]
    Set,
    #[strum(serialize = "frozenset")]
    FrozenSet,
    #[strum(serialize = "range")]
    Range,
    #[strum(serialize = "ellipsis")]
    Ellipsis,
    #[strum(serialize = "NotImplementedType")]
    NotImplemented,
}

/// Built-in input values; invalid payload contents are immaterial to division.
#[derive(Clone, Debug, PartialEq)]
pub enum BuiltinPriceValue {
    Bool(bool),
    Integer(BigInt),
    Float(f64),
    Complex(Complex<f64>),
    NonNumeric(BuiltinPriceType),
}

/// Python true division produces a float or complex, also for bool/int input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuiltinPriceNumber {
    Float(f64),
    Complex(Complex<f64>),
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum BuiltinPriceArithmeticError {
    #[error("int too large to convert to float")]
    IntegerOverflow,
    #[error("unsupported operand type(s) for /: '{0}' and 'float'")]
    UnsupportedDivision(BuiltinPriceType),
}

/// Concrete built-in adapter for the existing object ufunc stage engine.
pub struct BuiltinPriceArithmetic;

fn clean_scalar(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(f64::MIN, f64::MAX)
    }
}

impl ObjectPriceArithmetic for BuiltinPriceArithmetic {
    type Input = BuiltinPriceValue;
    type Quotient = BuiltinPriceNumber;
    type Difference = BuiltinPriceNumber;
    type Value = BuiltinPriceNumber;
    type Error = BuiltinPriceArithmeticError;

    fn divide(
        &mut self,
        value: &BuiltinPriceValue,
        baseline: f64,
    ) -> Result<BuiltinPriceNumber, Self::Error> {
        let value =
            match value {
                BuiltinPriceValue::Bool(value) => f64::from(u8::from(*value)),
                BuiltinPriceValue::Integer(value) => value
                    .to_f64()
                    .filter(|value| value.is_finite())
                    .ok_or(BuiltinPriceArithmeticError::IntegerOverflow)?,
                BuiltinPriceValue::Float(value) => *value,
                // CPython's built-in complex/float division is component-wise;
                // NumPy's complex-array ufunc follows a different operation order.
                BuiltinPriceValue::Complex(value) => {
                    return Ok(BuiltinPriceNumber::Complex(*value / baseline));
                }
                BuiltinPriceValue::NonNumeric(kind) => {
                    return Err(BuiltinPriceArithmeticError::UnsupportedDivision(*kind));
                }
            };
        Ok(BuiltinPriceNumber::Float(value / baseline))
    }

    fn subtract(
        &mut self,
        value: &BuiltinPriceNumber,
        buy: bool,
    ) -> Result<BuiltinPriceNumber, Self::Error> {
        Ok(match value {
            BuiltinPriceNumber::Float(value) => {
                BuiltinPriceNumber::Float(if buy { 1.0 - value } else { value - 1.0 })
            }
            BuiltinPriceNumber::Complex(value) => BuiltinPriceNumber::Complex(if buy {
                Complex::new(1.0 - value.re, -value.im)
            } else {
                Complex::new(value.re - 1.0, value.im)
            }),
        })
    }

    fn multiply(&mut self, value: &BuiltinPriceNumber) -> Result<BuiltinPriceNumber, Self::Error> {
        Ok(match value {
            BuiltinPriceNumber::Float(value) => BuiltinPriceNumber::Float(value * 10_000.0),
            BuiltinPriceNumber::Complex(value) => BuiltinPriceNumber::Complex(*value * 10_000.0),
        })
    }

    fn finish_scalar(
        &mut self,
        value: BuiltinPriceNumber,
    ) -> Result<BuiltinPriceNumber, Self::Error> {
        Ok(match value {
            BuiltinPriceNumber::Float(value) => BuiltinPriceNumber::Float(clean_scalar(value)),
            BuiltinPriceNumber::Complex(value) => BuiltinPriceNumber::Complex(Complex::new(
                clean_scalar(value.re),
                clean_scalar(value.im),
            )),
        })
    }
}

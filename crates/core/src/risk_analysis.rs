//! Source-compatible risk statistics from `qlib.contrib.evaluate.risk_analysis`.

use std::str::FromStr;

use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{Frequency, FrequencyError, FrequencyUnit};

/// The stable row order of Qlib's one-column `risk` result.
pub const RISK_ANALYSIS_FIELDS: [&str; 5] = [
    "mean",
    "std",
    "annualized_return",
    "information_ratio",
    "max_drawdown",
];

/// The five statistics returned by Qlib, in its observable row order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RiskAnalysisResult {
    /// Arithmetic mean in sum mode, or per-observation geometric mean in product mode.
    pub mean: f64,
    /// Sample standard deviation (`ddof=1`).
    pub std: f64,
    /// Return scaled to the selected annualization factor.
    pub annualized_return: f64,
    /// Mean divided by sample deviation and multiplied by the square root of the scaler.
    pub information_ratio: f64,
    /// Minimum drawdown of the accumulated curve.
    pub max_drawdown: f64,
}

impl RiskAnalysisResult {
    /// Return values in the same order as [`RISK_ANALYSIS_FIELDS`].
    #[must_use]
    pub const fn ordered_values(self) -> [f64; 5] {
        [
            self.mean,
            self.std,
            self.annualized_return,
            self.information_ratio,
            self.max_drawdown,
        ]
    }
}

/// Warning category exposed by the pure Rust compatibility boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskAnalysisWarningCategory {
    /// Explicit warning from Qlib's `N`/`freq` precedence rule.
    UserWarning,
    /// Numeric warning emitted by `NumPy` while evaluating special values.
    RuntimeWarning,
}

/// An ordered warning emitted while calculating risk statistics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskAnalysisWarning {
    /// Python-compatible warning category.
    pub category: RiskAnalysisWarningCategory,
    /// Python-compatible warning text.
    pub message: String,
}

/// Successful risk statistics and their ordered warnings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskAnalysisOutput {
    /// The `risk` column values.
    pub result: RiskAnalysisResult,
    /// Warnings in evaluation order.
    pub warnings: Vec<RiskAnalysisWarning>,
}

/// Source-compatible failures from risk analysis.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RiskAnalysisError {
    /// Neither annualization input was supplied.
    #[error("at least one of `N` and `freq` should exist")]
    MissingAnnualization,
    /// The existing Qlib frequency parser rejected the spelling.
    #[error(
        "freq format is not supported, the freq should be like (n)month/mon, (n)week/w, (n)day/d, (n)minute/min"
    )]
    InvalidFrequency {
        /// Detailed parser error retained as the error source.
        #[source]
        source: FrequencyError,
    },
    /// A zero frequency count makes the scaler division undefined.
    #[error("division by zero")]
    ZeroFrequency,
    /// Product mode accesses the last cumulative observation positionally.
    #[error("single positional indexer is out-of-bounds")]
    EmptyProduct,
    /// The requested accumulation mode is not one of Qlib's two modes.
    #[error(
        "risk_analysis accumulation mode {mode} is not supported. Expected `sum` or `product`."
    )]
    UnsupportedMode {
        /// Rejected mode.
        mode: String,
    },
}

/// Calculate Qlib-compatible risk statistics over a float return series.
///
/// `NaN` represents a missing Pandas value. The total slice length remains observable in
/// product-mode exponents, while reductions skip missing values just as Pandas does.
///
/// # Errors
///
/// Returns a typed error for missing/invalid annualization, a zero frequency count, an empty
/// product series, or an unsupported mode.
pub fn risk_analysis(
    returns: &[f64],
    n: Option<f64>,
    frequency: Option<&str>,
    mode: &str,
) -> Result<RiskAnalysisOutput, RiskAnalysisError> {
    let mut warnings = Vec::new();
    let scaler = match (n, frequency) {
        (None, None) => return Err(RiskAnalysisError::MissingAnnualization),
        (Some(value), Some(_)) => {
            warnings.push(RiskAnalysisWarning {
                category: RiskAnalysisWarningCategory::UserWarning,
                message: "risk_analysis freq will be ignored".to_owned(),
            });
            value
        }
        (Some(value), None) => value,
        (None, Some(value)) => annualization_scaler(value)?,
    };

    let result = match mode {
        "sum" => sum_analysis(returns, scaler, &mut warnings),
        "product" => product_analysis(returns, scaler, &mut warnings)?,
        _ => {
            return Err(RiskAnalysisError::UnsupportedMode {
                mode: mode.to_owned(),
            });
        }
    };
    Ok(RiskAnalysisOutput { result, warnings })
}

fn annualization_scaler(value: &str) -> Result<f64, RiskAnalysisError> {
    let frequency = Frequency::from_str(value)
        .map_err(|source| RiskAnalysisError::InvalidFrequency { source })?;
    if frequency.count.is_zero() {
        return Err(RiskAnalysisError::ZeroFrequency);
    }
    let periods = match frequency.unit {
        FrequencyUnit::Minute => 240.0 * 238.0,
        FrequencyUnit::Day => 238.0,
        FrequencyUnit::Week => 50.0,
        FrequencyUnit::Month => 12.0,
    };
    Ok(frequency
        .count
        .to_f64()
        .map_or(0.0, |count| periods / count))
}

fn sum_analysis(
    returns: &[f64],
    scaler: f64,
    warnings: &mut Vec<RiskAnalysisWarning>,
) -> RiskAnalysisResult {
    let mean = pandas_mean(returns);
    if reduction_overflowed(returns, mean) {
        runtime_warning(warnings, "overflow encountered in reduce");
    } else if has_opposite_infinities(returns) {
        runtime_warning(warnings, "invalid value encountered in reduce");
    }
    let std = pandas_sample_std(returns, mean);
    numeric_std_warning(returns, mean, warnings);

    let max_drawdown = additive_drawdown(returns);
    if cumulative_overflow(returns, 0.0, |left, right| left + right) {
        runtime_warning(warnings, "overflow encountered in accumulate");
        runtime_warning(warnings, "overflow encountered in accumulate");
    }
    if cumulative_invalid(returns, 0.0, |left, right| left + right) {
        runtime_warning(warnings, "invalid value encountered in accumulate");
        runtime_warning(warnings, "invalid value encountered in accumulate");
    }
    scalar_information_warnings(mean, std, scaler, warnings);
    let information_ratio = mean / std * scaler.sqrt();
    RiskAnalysisResult {
        mean,
        std,
        annualized_return: mean * scaler,
        information_ratio,
        max_drawdown,
    }
}

fn product_analysis(
    returns: &[f64],
    scaler: f64,
    warnings: &mut Vec<RiskAnalysisWarning>,
) -> Result<RiskAnalysisResult, RiskAnalysisError> {
    if returns.is_empty() {
        return Err(RiskAnalysisError::EmptyProduct);
    }
    let cumulative = cumulative_values(returns, 1.0, |state, value| state * (1.0 + value));
    numeric_cumulative_product_warning(returns, warnings);
    let final_value = cumulative[cumulative.len() - 1];
    let observation_count = returns
        .len()
        .to_f64()
        .expect("slice lengths are representable as f64 on supported targets");
    let mean_exponent = 1.0 / observation_count;
    if final_value.is_finite() && final_value.is_sign_negative() && !is_integer(mean_exponent) {
        runtime_warning(warnings, "invalid value encountered in scalar power");
    }
    let mean = final_value.powf(mean_exponent) - 1.0;

    let mut log_returns = Vec::with_capacity(returns.len());
    let mut saw_zero = false;
    let mut saw_negative = false;
    for value in returns {
        let gross = 1.0 + value;
        saw_zero |= gross == 0.0;
        saw_negative |= gross < 0.0;
        log_returns.push(gross.ln());
    }
    if saw_zero {
        runtime_warning(warnings, "divide by zero encountered in log");
    }
    if saw_negative {
        runtime_warning(warnings, "invalid value encountered in log");
    }
    let log_mean = pandas_mean(&log_returns);
    let std = pandas_sample_std(&log_returns, log_mean);
    numeric_std_warning(&log_returns, log_mean, warnings);

    let annual_exponent = scaler / observation_count;
    if final_value.is_finite()
        && final_value.is_sign_negative()
        && annual_exponent.is_finite()
        && !is_integer(annual_exponent)
    {
        runtime_warning(warnings, "invalid value encountered in scalar power");
    }
    let cumulative_return = final_value - 1.0;
    let annual_base = 1.0 + cumulative_return;
    let annualized_return = annual_base.powf(annual_exponent) - 1.0;
    let max_drawdown = multiplicative_drawdown(&cumulative);
    scalar_information_warnings(mean, std, scaler, warnings);
    Ok(RiskAnalysisResult {
        mean,
        std,
        annualized_return,
        information_ratio: mean / std * scaler.sqrt(),
        max_drawdown,
    })
}

fn pandas_mean(values: &[f64]) -> f64 {
    let replaced: Vec<_> = values
        .iter()
        .map(|value| if value.is_nan() { 0.0 } else { *value })
        .collect();
    let count = values.iter().filter(|value| !value.is_nan()).count();
    if count == 0 {
        f64::NAN
    } else {
        numpy_pairwise_sum(&replaced)
            / count
                .to_f64()
                .expect("slice lengths are representable as f64 on supported targets")
    }
}

fn pandas_sample_std(values: &[f64], mean: f64) -> f64 {
    let deviations: Vec<_> = values
        .iter()
        .map(|value| {
            if value.is_nan() {
                0.0
            } else {
                let deviation = *value - mean;
                deviation * deviation
            }
        })
        .collect();
    let count = values.iter().filter(|value| !value.is_nan()).count();
    if count <= 1 {
        f64::NAN
    } else {
        (numpy_pairwise_sum(&deviations)
            / (count - 1)
                .to_f64()
                .expect("slice lengths are representable as f64 on supported targets"))
        .sqrt()
    }
}

// NumPy uses this eight-lane, 128-value-block pairwise reduction for contiguous f64 arrays.
fn numpy_pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK_SIZE: usize = 128;
    if values.len() <= 7 {
        values.iter().fold(-0.0, |sum, value| sum + value)
    } else if values.len() <= BLOCK_SIZE {
        let mut lanes = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let mut offset = 8;
        while offset + 7 < values.len() {
            for lane in 0..8 {
                lanes[lane] += values[offset + lane];
            }
            offset += 8;
        }
        let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for value in &values[offset..] {
            sum += value;
        }
        sum
    } else {
        let mut midpoint = values.len() / 2;
        midpoint -= midpoint % 8;
        numpy_pairwise_sum(&values[..midpoint]) + numpy_pairwise_sum(&values[midpoint..])
    }
}

fn cumulative_values(
    values: &[f64],
    initial: f64,
    operation: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let mut state = initial;
    values
        .iter()
        .map(|value| {
            if value.is_nan() {
                f64::NAN
            } else {
                state = operation(state, *value);
                state
            }
        })
        .collect()
}

fn additive_drawdown(values: &[f64]) -> f64 {
    let cumulative = cumulative_values(values, 0.0, |state, value| state + value);
    drawdown_min(&cumulative, |value, peak| value - peak)
}

fn multiplicative_drawdown(cumulative: &[f64]) -> f64 {
    drawdown_min(cumulative, |value, peak| value / peak - 1.0)
}

fn drawdown_min(cumulative: &[f64], difference: impl Fn(f64, f64) -> f64) -> f64 {
    let mut peak: Option<f64> = None;
    let mut minimum: Option<f64> = None;
    for value in cumulative.iter().copied().filter(|value| !value.is_nan()) {
        peak = Some(peak.map_or(value, |current| current.max(value)));
        let drawdown = difference(value, peak.expect("a peak was just assigned"));
        if !drawdown.is_nan() {
            minimum = Some(minimum.map_or(drawdown, |current| current.min(drawdown)));
        }
    }
    minimum.unwrap_or(f64::NAN)
}

fn has_opposite_infinities(values: &[f64]) -> bool {
    values.contains(&f64::INFINITY) && values.contains(&f64::NEG_INFINITY)
}

fn numeric_std_warning(values: &[f64], mean: f64, warnings: &mut Vec<RiskAnalysisWarning>) {
    if has_opposite_infinities(values) {
        runtime_warning(warnings, "invalid value encountered in reduce");
    } else if reduction_overflowed(values, mean) {
        runtime_warning(warnings, "overflow encountered in reduce");
    } else if mean.is_infinite() {
        runtime_warning(warnings, "invalid value encountered in subtract");
    }
}

fn scalar_information_warnings(
    mean: f64,
    std: f64,
    scaler: f64,
    warnings: &mut Vec<RiskAnalysisWarning>,
) {
    if std == 0.0 && mean.is_finite() {
        let message = if mean == 0.0 {
            "invalid value encountered in scalar divide"
        } else {
            "divide by zero encountered in scalar divide"
        };
        runtime_warning(warnings, message);
    } else if mean.is_infinite() && std.is_infinite() {
        runtime_warning(warnings, "invalid value encountered in scalar divide");
    }
    if scaler.is_sign_negative() {
        runtime_warning(warnings, "invalid value encountered in sqrt");
    }
}

fn reduction_overflowed(values: &[f64], result: f64) -> bool {
    result.is_infinite()
        && values
            .iter()
            .all(|value| value.is_finite() || value.is_nan())
}

fn numeric_cumulative_product_warning(returns: &[f64], warnings: &mut Vec<RiskAnalysisWarning>) {
    let mut state = 1.0;
    let mut overflow = false;
    let mut invalid = false;
    for value in returns.iter().copied().filter(|value| !value.is_nan()) {
        let gross = 1.0 + value;
        let next = state * gross;
        overflow |= next.is_infinite() && state.is_finite() && gross.is_finite();
        invalid |= next.is_nan() && !state.is_nan();
        state = next;
    }
    if overflow {
        runtime_warning(warnings, "overflow encountered in accumulate");
    }
    if invalid {
        runtime_warning(warnings, "invalid value encountered in accumulate");
    }
}

fn cumulative_invalid(values: &[f64], initial: f64, operation: impl Fn(f64, f64) -> f64) -> bool {
    let mut state = initial;
    for value in values.iter().copied().filter(|value| !value.is_nan()) {
        let next = operation(state, value);
        if next.is_nan() {
            return true;
        }
        state = next;
    }
    false
}

fn cumulative_overflow(values: &[f64], initial: f64, operation: impl Fn(f64, f64) -> f64) -> bool {
    let mut state = initial;
    for value in values.iter().copied().filter(|value| !value.is_nan()) {
        let next = operation(state, value);
        if next.is_infinite() && state.is_finite() && value.is_finite() {
            return true;
        }
        state = next;
    }
    false
}

fn is_integer(value: f64) -> bool {
    value.fract() == 0.0
}

fn runtime_warning(warnings: &mut Vec<RiskAnalysisWarning>, message: &'static str) {
    warnings.push(RiskAnalysisWarning {
        category: RiskAnalysisWarningCategory::RuntimeWarning,
        message: message.to_owned(),
    });
}

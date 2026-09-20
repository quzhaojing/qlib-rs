//! Scalar formatting for Checkpoint filename adapters, using third-party format engines.
//!
//! This is the value-formatting layer, not the replacement-field parser. Locale `n` uses
//! the C numeric locale (the Python startup default), not mutable process-global locales.

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use regex::Regex;
use rustpython_format::FormatSpec;
use std::sync::LazyLock;

use crate::TrainingMetricScalar;
use crate::rl_checkpoint_numeric::normalize_numeric_fields;

fn pyformat(result: Result<String, pyformat_rs::FormatError>) -> Result<String, String> {
    result.map_err(|error| error.to_string())
}

fn integer(value: &BigInt, spec: &str) -> Result<String, String> {
    if spec.ends_with(['e', 'E', 'f', 'F', 'g', 'G', '%']) {
        // Validate as an integer before conversion. RustPython's large integer float
        // presentation can emit infinity instead of Python's conversion overflow error.
        float(0.0, spec)?;
        let number = value
            .to_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| "int too large to convert to float".to_owned())?;
        return float(number, spec);
    }
    if spec.ends_with('c') {
        // Only Unicode scalar values can be represented by Rust String. pyformat checks
        // that range without RustPython's surrogate-code-point panic. Integers outside
        // i128 are necessarily outside the character range as well.
        return pyformat(pyformat_rs::format_int(value.to_i128().unwrap_or(-1), spec));
    }
    if spec.starts_with('!') {
        // RustPython also parses conversion prefixes inside a mini-language spec, whereas
        // CPython only permits them in the surrounding replacement field. Keep valid '!'
        // fill characters, and let pyformat reject misplaced conversions.
        pyformat(pyformat_rs::format_int(0, spec))?;
    }
    let format = FormatSpec::parse(spec).map_err(|error| format!("{error:?}"))?;
    // The dependency uses num-bigint 0.4, while Qlib uses 0.5. Decimal interchange retains
    // arbitrary precision; this cannot fail for the decimal spelling of an existing BigInt.
    let number = value.to_string().parse().expect("BigInt decimal is valid");
    format
        .format_int(&number)
        .map_err(|error| format!("{error:?}"))
}

fn float(value: f64, spec: &str) -> Result<String, String> {
    if let Some(prefix) = spec.strip_suffix('n') {
        // In the C locale `n` is `g` without explicit grouping. Permit a comma/underscore
        // fill character, but not grouping. Delegate all other parsing (including `z`),
        // rounding and padding to pyformat; RustPython does not support `z` with `n`.
        let mut chars = prefix.chars();
        chars.next();
        let body = if matches!(chars.next(), Some('<' | '>' | '=' | '^')) {
            chars.as_str()
        } else {
            prefix
        };
        if body.contains([',', '_']) {
            return Err("cannot specify grouping with 'n'".into());
        }
        return floating_engine(value, &format!("{prefix}g"));
    }
    floating_engine(value, spec)
}

fn floating_engine(value: f64, spec: &str) -> Result<String, String> {
    // Both format engines keep their parsed fields private. Capture only the
    // supported large-precision shape; the engine still validates and renders it.
    static LARGE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
        r"(?s)\A(?P<layout>.[<>=^]|[<>=^])?(?P<flags>[+ -]?z?#?)(?P<zero>0?)(?P<width>[0-9]*)(?P<group>[,_]?)\.(?P<p>[0-9]+)(?P<kind>[eEfFgG%]?)\z"
    ).expect("large precision pattern is valid")
    });
    let Some(parts) = LARGE.captures(spec) else {
        return pyformat(pyformat_rs::format_float(value, spec));
    };
    let precision = parts["p"]
        .parse::<usize>()
        .expect("numeric fields were range-checked at the scalar boundary");
    if precision <= 9999 {
        return pyformat(pyformat_rs::format_float(value, spec));
    }
    if precision > i32::MAX as usize {
        return Err("float precision too large".into());
    }
    // Every finite f64 is an exact terminating decimal within 1,074 fractional
    // places; 1,100 also exceeds its maximum significant digit count. Larger
    // precisions cannot change rounding or the g/no-type exponent decision.
    let layout = parts.name("layout").map_or("", |part| part.as_str());
    let flags = &parts["flags"];
    let zero = &parts["zero"];
    let group = &parts["group"];
    let kind = &parts["kind"];
    let width = if parts["width"].is_empty() {
        0
    } else {
        parts["width"]
            .parse::<usize>()
            .expect("numeric fields were range-checked at the scalar boundary")
    };
    let nonfinite = !value.is_finite() || (kind == "%" && !(value * 100.0).is_finite());
    let trimmed = matches!(kind, "g" | "G" | "") && !flags.contains('#');
    if nonfinite || trimmed {
        return Ok(canonical_float(
            value,
            &format!("{layout}{flags}{zero}{width}{group}.1100{kind}"),
        ));
    }
    let extra = precision - 1100;
    let reduced_width = width.saturating_sub(extra);
    let body = canonical_float(value, &format!("{flags}{group}.1100{kind}"));
    let mut output = canonical_float(
        value,
        &format!("{layout}{flags}{zero}{reduced_width}{group}.1100{kind}"),
    );
    // The engine supplies all padding and sign-aware grouped zero fill. Reduce
    // width by the added tail, then insert before exponent/percent/trailing fill.
    let padding = reduced_width.saturating_sub(body.len());
    let trailing = match layout.chars().last() {
        Some('<') => padding,
        Some('^') => padding - padding / 2,
        _ => 0,
    };
    let suffix = body.len() - body.find(['e', 'E', '%']).unwrap_or(body.len());
    let position = output.chars().count() - trailing - suffix;
    let offset = output
        .char_indices()
        .nth(position)
        .map_or(output.len(), |(offset, _)| offset);
    output.insert_str(offset, &"0".repeat(extra));
    Ok(output)
}

fn canonical_float(value: f64, spec: &str) -> String {
    // Only called with the regex's supported float types/flags, prevalidated
    // widths, and fixed precision 1,100. Failure would be a pinned-engine contract
    // violation, not a recoverable user format error. User input uses Result.
    pyformat_rs::format_float(value, spec).expect("canonical bounded float format is valid")
}

pub(crate) fn validate_digit_runs(spec: &str) -> Result<(), String> {
    // pyformat's width/precision accumulator is unchecked. Reject overflowing
    // numeric fields before entering that dependency, including malformed specs.
    for digits in spec
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
    {
        digits
            .parse::<usize>()
            .ok()
            .filter(|value| isize::try_from(*value).is_ok())
            .ok_or("format numeric field is too large")?;
    }
    Ok(())
}

/// Format a primitive metric using Python's numeric/string mini-language in the C locale.
/// Filename field lookup, nested replacement fields, conversions and custom model object
/// protocols belong to the caller's filename adapter, not this scalar layer.
///
/// # Errors
/// Returns invalid format/type/conversion errors. Custom display-only values cannot provide
/// a Python `__format__` protocol and must use an explicit model-aware filename adapter.
pub fn format_rl_checkpoint_scalar(
    value: &TrainingMetricScalar,
    spec: &str,
) -> Result<String, String> {
    let normalized = normalize_numeric_fields(spec)?;
    let spec = normalized.as_ref();
    validate_digit_runs(spec)?;
    fractional_scalar(value, spec)
}

fn fractional_scalar(value: &TrainingMetricScalar, spec: &str) -> Result<String, String> {
    // Python 3.14 adds a fractional separator after optional precision. Neither
    // pinned engine supports it; keep conversion, integer grouping and padding
    // delegated, adding only the forward-grouped fractional digits.
    static FRACTIONAL: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?s)\A(?P<layout>.[<>=^]|[<>=^])?(?P<flags>[+ -]?z?#?)(?P<zero>0?)(?P<width>[0-9]*)(?P<group>[,_]?)\.(?P<p>[0-9]*)(?P<separator>[,_])(?P<kind>[a-zA-Z%]?)\z",
        )
        .expect("fractional grouping pattern is valid")
    });
    let Some(parts) = FRACTIONAL.captures(spec) else {
        return scalar_engine(value, spec);
    };
    let kind = &parts["kind"];
    if kind == "n" {
        return Err("cannot specify fractional grouping with 'n'".into());
    }
    let layout = parts.name("layout").map_or("", |part| part.as_str());
    let flags = &parts["flags"];
    let zero = &parts["zero"];
    let group = &parts["group"];
    let precision = &parts["p"];
    let precision = if precision.is_empty() {
        String::new()
    } else {
        format!(".{precision}")
    };
    let width = if parts["width"].is_empty() {
        "0"
    } else {
        &parts["width"]
    };
    // Always retain a nonempty spec: `format(True, '._')` is '1', not 'True',
    // and None must still reject it. A zero width has no other rendering effect.
    let plain = format!("{layout}{flags}{zero}{width}{group}{precision}{kind}");
    let floating = matches!(value, TrainingMetricScalar::Float(_))
        || (matches!(
            value,
            TrainingMetricScalar::Integer(_) | TrainingMetricScalar::Boolean(_)
        ) && kind.ends_with(['e', 'E', 'f', 'F', 'g', 'G', '%']));
    if !floating {
        return scalar_engine(value, &plain);
    }
    let body = scalar_engine(value, &format!("{flags}0{group}{precision}{kind}"))?;
    let fraction_start = body.find('.').map_or(body.len(), |index| index + 1);
    let digits = &body[fraction_start..];
    let fraction_len = digits.bytes().take_while(u8::is_ascii_digit).count();
    let extra = fraction_len.saturating_sub(1) / 3;
    if extra == 0 {
        return scalar_engine(value, &plain);
    }
    let width = width
        .parse::<usize>()
        .expect("numeric fields were range-checked at the scalar boundary");
    let reduced_width = width.saturating_sub(extra);
    // The body call validated this primitive value, precision, flags, grouping
    // and type. The regex admits only valid numeric alignments/fills, and width
    // only shrinks within its checked range. Re-rendering this canonical layout
    // is therefore an engine invariant, not another fallible user-input boundary.
    let mut output = scalar_engine(
        value,
        &format!("{layout}{flags}{zero}{reduced_width}{group}{precision}{kind}"),
    )
    .expect("validated numeric body accepts canonical reduced-width layout");
    let padding = reduced_width.saturating_sub(body.len());
    let trailing = match layout.chars().last() {
        Some('<') => padding,
        Some('^') => padding - padding / 2,
        _ => 0,
    };
    let suffix = body.len() - fraction_start - fraction_len;
    let end = output.chars().count() - trailing - suffix;
    let start = end - fraction_len;
    // Locate by numeric-body lengths, never by searching the padded output:
    // literal fill characters can themselves be '.', digits, 'e' or '%'.
    let start = output
        .char_indices()
        .nth(start)
        .expect("nonempty fractional digits occur inside the formatted output")
        .0;
    let end = start + fraction_len; // Fractional digits are ASCII.
    let grouped = (0..fraction_len)
        .step_by(3)
        .map(|index| &digits[index..(index + 3).min(fraction_len)])
        .collect::<Vec<_>>()
        .join(&parts["separator"]);
    output.replace_range(start..end, &grouped);
    Ok(output)
}

fn scalar_engine(value: &TrainingMetricScalar, spec: &str) -> Result<String, String> {
    match value {
        TrainingMetricScalar::Integer(value) => integer(value, spec),
        TrainingMetricScalar::Float(value) => float(*value, spec),
        TrainingMetricScalar::Text(value) => pyformat(pyformat_rs::format_str(value, spec)),
        TrainingMetricScalar::Boolean(value) => {
            if spec.is_empty() {
                Ok(if *value { "True" } else { "False" }.into())
            } else {
                integer(&BigInt::from(u8::from(*value)), spec)
            }
        }
        TrainingMetricScalar::Null => {
            if spec.is_empty() {
                Ok("None".into())
            } else {
                Err("unsupported format string passed to NoneType.__format__".into())
            }
        }
        TrainingMetricScalar::Custom(_) => {
            Err("custom checkpoint values require a model-aware filename adapter".into())
        }
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_checkpoint_format.rs"]
mod tests;

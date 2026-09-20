//! Python numeric locale formatting over explicit, per-field locale snapshots.

use regex::Regex;
use std::sync::LazyLock;
use thousands::{Separable, SeparatorPolicy, digits};

use crate::rl_checkpoint_format::validate_digit_runs;
use crate::rl_checkpoint_numeric::normalize_numeric_fields;
use crate::{TrainingMetricScalar, format_rl_checkpoint_scalar};

/// Numeric fields of `localeconv`, independent of mutable global C locale state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RlCheckpointNumericLocale {
    decimal: String,
    separator: String,
    groups: Vec<u8>,
    stop: bool,
}

impl Default for RlCheckpointNumericLocale {
    fn default() -> Self {
        Self {
            decimal: ".".into(),
            separator: String::new(),
            groups: Vec::new(),
            stop: false,
        }
    }
}

impl RlCheckpointNumericLocale {
    /// Construct from decoded locale metadata. Zero repeats the last size;
    /// `CHAR_MAX` (127) stops grouping. Entries after either terminator are ignored.
    /// # Errors
    /// Rejects empty decimal marks and invalid grouping sizes before using the library.
    pub fn new(decimal: String, separator: String, grouping: &[u8]) -> Result<Self, String> {
        if decimal.is_empty() {
            return Err("locale decimal mark must not be empty".into());
        }
        let mut result = Self {
            decimal,
            separator,
            groups: Vec::new(),
            stop: false,
        };
        for &size in grouping {
            match size {
                0 => break,
                127 => {
                    result.stop = true;
                    break;
                }
                1..=126 => result.groups.push(size),
                _ => return Err("locale grouping size exceeds CHAR_MAX".into()),
            }
        }
        Ok(result)
    }

    fn grouped(&self, value: &str) -> String {
        let policy = SeparatorPolicy {
            separator: &self.separator,
            groups: &self.groups,
            digits: digits::ASCII_DECIMAL,
        };
        if self.stop && !self.groups.is_empty() {
            // The library repeats its final group. For a stop terminator, leave
            // the ungrouped prefix outside it and delegate only the bounded tail.
            let tail: usize = self.groups.iter().map(|&n| usize::from(n)).sum();
            let start = value.len().saturating_sub(tail);
            if start > 0 {
                return format!(
                    "{}{}{}",
                    &value[..start],
                    self.separator,
                    value[start..].separate_by_policy(policy)
                );
            }
        }
        value.separate_by_policy(policy)
    }

    fn zero_grouped(&self, value: &str, width: usize) -> String {
        // Grouped length is monotone in the number of leading zeros. Search the
        // minimum using the actual library output, including multi-scalar separators.
        let mut lower = value.len();
        let mut upper = width.max(lower);
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            let candidate = format!("{}{}", "0".repeat(middle - value.len()), value);
            if self.grouped(&candidate).chars().count() < width {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        self.grouped(&format!("{}{}", "0".repeat(lower - value.len()), value))
    }
}

/// Query at a numeric field's formatting position, not at template construction.
/// Implementations own locale synchronization and decoding. A snapshot itself is
/// a constant provider; this trait does not silently read the OS default language.
pub trait RlCheckpointLocaleProvider: Send + Sync {
    /// # Errors
    /// Propagates locale acquisition/decoding failures before rendering that field.
    fn numeric_locale(&self) -> Result<RlCheckpointNumericLocale, String>;
}

impl RlCheckpointLocaleProvider for RlCheckpointNumericLocale {
    fn numeric_locale(&self) -> Result<RlCheckpointNumericLocale, String> {
        Ok(self.clone())
    }
}

/// Format a primitive with a locale provider. Other presentations remain locale-independent.
/// # Errors
/// Returns invalid formats, unsupported value types, or locale-provider failures.
/// # Panics
/// Panics if the pinned padding engine violates a validated canonical layout invariant.
pub fn format_rl_checkpoint_scalar_with_locale(
    value: &TrainingMetricScalar,
    spec: &str,
    provider: &dyn RlCheckpointLocaleProvider,
) -> Result<String, String> {
    static LOCALE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?s)\A(?P<layout>.[<>=^]|[<>=^])?(?P<flags>[+ -]?z?#?)(?P<zero>0?)(?P<width>[0-9]*)(?P<precision>\.[0-9]+)?n\z")
            .expect("numeric locale pattern is valid")
    });
    if !spec.ends_with('n') {
        return format_rl_checkpoint_scalar(value, spec);
    }
    let normalized = normalize_numeric_fields(spec)?;
    let spec = normalized.as_ref();
    validate_digit_runs(spec)?;
    let Some(parts) = LOCALE.captures(spec) else {
        return format_rl_checkpoint_scalar(value, spec);
    };
    let flags = &parts["flags"];
    let precision = parts.name("precision").map_or("", |p| p.as_str());
    let body = format_rl_checkpoint_scalar(value, &format!("{flags}0{precision}n"))?;
    let locale = provider.numeric_locale()?;
    let sign_len = usize::from(body.starts_with(['+', '-', ' ']));
    let integer_len = body[sign_len..]
        .bytes()
        .take_while(u8::is_ascii_digit)
        .count();
    if integer_len == 0 {
        // Python keeps ASCII inf/nan and never groups their zero padding.
        return format_rl_checkpoint_scalar(value, spec);
    }
    let (sign, number) = body.split_at(sign_len);
    let (integer, remainder) = number.split_at(integer_len);
    let remainder = remainder.replacen('.', &locale.decimal, 1);
    let width = parts["width"].parse::<usize>().unwrap_or(0);
    let layout = parts.name("layout").map_or("", |p| p.as_str());
    let mut chars = layout.chars();
    let first = chars.next();
    let second = chars.next();
    let zero = !parts["zero"].is_empty();
    let fill = second.map_or(if zero { '0' } else { ' ' }, |_| {
        first.expect("two-character layout has a fill")
    });
    let align = second.or(first).unwrap_or(if zero { '=' } else { '>' });
    let grouped = if align == '=' && fill == '0' {
        locale.zero_grouped(
            integer,
            width.saturating_sub(sign_len + remainder.chars().count()),
        )
    } else {
        locale.grouped(integer)
    };
    let number = format!("{grouped}{remainder}");
    // Delegate padding to the existing string engine; '=' pads after the ASCII
    // sign, while other alignments include it in the padded text.
    if align == '=' {
        let padded = pyformat_rs::format_str(
            &number,
            &format!("{fill}>{}", width.saturating_sub(sign_len)),
        )
        .expect("canonical string padding is valid");
        Ok(format!("{sign}{padded}"))
    } else {
        Ok(
            pyformat_rs::format_str(&format!("{sign}{number}"), &format!("{fill}{align}{width}"))
                .expect("canonical string padding is valid"),
        )
    }
}

#[cfg(test)]
#[path = "../tests/support/rl_checkpoint_locale.rs"]
mod tests;

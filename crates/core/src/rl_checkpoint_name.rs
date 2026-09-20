//! Streaming Python-style checkpoint names over explicit model value protocols.

use std::{fmt::Write, sync::Arc};

use indexmap::IndexMap;
use num_bigint::BigInt;
use rustpython_literal::escape::{Escape, Quote, UnicodeEscape};

use crate::rl_checkpoint_numeric::decimal_digit;
use crate::rl_checkpoint_template::{field, part};
use crate::{
    RlCheckpointLocaleProvider, RlCheckpointName, RlCheckpointNumericLocale, TrainingMetricScalar,
    format_rl_checkpoint_scalar, format_rl_checkpoint_scalar_with_locale,
};

/// Python distinguishes unquoted decimal integer subscripts from literal string keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RlCheckpointFieldIndex<'a> {
    Integer(usize),
    Key(&'a str),
}

fn integer_index(text: &str) -> Result<Option<usize>, String> {
    if text.is_empty() {
        return Ok(None);
    }
    let mut value = 0_usize;
    for ch in text.chars() {
        let Some(digit) = decimal_digit(ch) else {
            return Ok(None);
        };
        value = value
            .checked_mul(10)
            .and_then(|n| n.checked_add(digit))
            .filter(|n| isize::try_from(*n).is_ok())
            .ok_or("too many checkpoint index digits")?;
    }
    Ok(Some(value))
}

/// Model-owned behavior used by a replacement field. Objects are borrowed at the root;
/// lookup results can retain shared model identity through Arc instead of copying state.
/// No method is called while a Trainer lock is held. These are linked Rust plugins, not ABI.
pub trait RlCheckpointFormatValue: Send + Sync {
    /// Python `__format__` behavior.
    /// # Errors
    /// Returns a value/format failure at this field's original evaluation position.
    fn format(&self, spec: &str) -> Result<String, String>;
    /// Format with the surrounding numeric-locale context. Custom model protocols
    /// retain control by default; primitive and wrapper implementations may use it.
    /// # Errors
    /// Returns the model format failure or a numeric locale acquisition failure.
    fn format_with_locale(
        &self,
        spec: &str,
        _locale: &dyn RlCheckpointLocaleProvider,
    ) -> Result<String, String> {
        self.format(spec)
    }
    /// Python `str` or `repr`; ASCII conversion is applied by the adapter after repr.
    /// # Errors
    /// Returns conversion failure before any nested specification is evaluated.
    fn representation(&self, repr: bool) -> Result<String, String>;
    /// Python attribute lookup.
    /// # Errors
    /// Returns a missing/unsupported attribute or model lookup failure.
    fn attribute(&self, name: &str) -> Result<Arc<dyn RlCheckpointFormatValue>, String>;
    /// Python item lookup. Unicode decimal indices have already been distinguished from
    /// literal string keys; negative signs, quotes and colons remain part of string keys.
    /// # Errors
    /// Returns a missing/unsupported item or model lookup failure.
    fn item(
        &self,
        key: RlCheckpointFieldIndex<'_>,
    ) -> Result<Arc<dyn RlCheckpointFormatValue>, String>;
}

// Shared trait objects let a metric table hold heterogeneous, non-Clone models.
// Forward through the pointee without cloning state or dropping locale context.
impl<T: RlCheckpointFormatValue + ?Sized> RlCheckpointFormatValue for Arc<T> {
    fn format(&self, spec: &str) -> Result<String, String> {
        self.as_ref().format(spec)
    }
    fn format_with_locale(
        &self,
        spec: &str,
        locale: &dyn RlCheckpointLocaleProvider,
    ) -> Result<String, String> {
        self.as_ref().format_with_locale(spec, locale)
    }
    fn representation(&self, repr: bool) -> Result<String, String> {
        self.as_ref().representation(repr)
    }
    fn attribute(&self, name: &str) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        self.as_ref().attribute(name)
    }
    fn item(
        &self,
        key: RlCheckpointFieldIndex<'_>,
    ) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        self.as_ref().item(key)
    }
}

impl RlCheckpointFormatValue for TrainingMetricScalar {
    fn format(&self, spec: &str) -> Result<String, String> {
        format_rl_checkpoint_scalar(self, spec)
    }
    fn format_with_locale(
        &self,
        spec: &str,
        locale: &dyn RlCheckpointLocaleProvider,
    ) -> Result<String, String> {
        format_rl_checkpoint_scalar_with_locale(self, spec, locale)
    }
    fn representation(&self, repr: bool) -> Result<String, String> {
        if let Self::Text(text) = self {
            return Ok(if repr {
                python_string_repr(text)
            } else {
                text.clone()
            });
        }
        self.format("")
    }
    fn attribute(&self, name: &str) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        let result = match (self, name) {
            (Self::Integer(_) | Self::Float(_), "real") | (Self::Integer(_), "numerator") => {
                self.clone()
            }
            (Self::Integer(_) | Self::Boolean(_), "imag") => Self::Integer(0.into()),
            (Self::Float(_), "imag") => Self::Float(0.0),
            (Self::Integer(_) | Self::Boolean(_), "denominator") => Self::Integer(1.into()),
            (Self::Boolean(value), "real" | "numerator") => Self::Integer(u8::from(*value).into()),
            _ => return Err(format!("unsupported checkpoint attribute {name}")),
        };
        Ok(Arc::new(result))
    }
    fn item(
        &self,
        key: RlCheckpointFieldIndex<'_>,
    ) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        if let (Self::Text(text), RlCheckpointFieldIndex::Integer(index)) = (self, key) {
            let value = text
                .chars()
                .nth(index)
                .ok_or("checkpoint string index out of range")?;
            return Ok(Arc::new(Self::Text(value.to_string())));
        }
        Err(format!("unsupported checkpoint item {key:?}"))
    }
}

// Keep the dependency's global quote selection and ASCII escaping, but use the
// oracle's Unicode version for non-ASCII printability. RustPython's older table
// escapes newly assigned printable characters (18,289 exhaustive differences).
pub(crate) fn python_string_repr(text: &str) -> String {
    python_code_points_repr(text.chars().map(u32::from))
}

// Callers supply validated Python code points, which may include lone surrogates.
// Surrogates do not affect quote choice. Escape them at their original position,
// never by replacing marker escape strings in the already-escaped output.
pub(crate) fn python_code_points_repr(points: impl Iterator<Item = u32> + Clone) -> String {
    use intl::unicode::{Group, age, general_category};
    let scalar_text: String = points.clone().filter_map(char::from_u32).collect();
    let quote = UnicodeEscape::repr_layout(&scalar_text, Quote::Single).quote;
    let mut output = String::new();
    output.push(quote.to_char());
    for point in points {
        let Some(ch) = char::from_u32(point) else {
            write!(output, "\\u{point:04x}").expect("writing into String cannot fail");
            continue;
        };
        if ch.is_ascii() {
            UnicodeEscape::with_forced_quote(&ch.to_string(), quote)
                .write_body(&mut output)
                .expect("writing into String cannot fail");
        } else if age(ch).is_some_and(|version| version <= (16, 0))
            && !matches!(
                general_category(ch).group(),
                Group::Other | Group::Separator
            )
        {
            output.push(ch);
        } else {
            output.push_str(&ascii_repr(&ch.to_string()));
        }
    }
    output.push(quote.to_char());
    output
}

impl RlCheckpointFormatValue for f64 {
    fn format(&self, spec: &str) -> Result<String, String> {
        TrainingMetricScalar::Float(*self).format(spec)
    }
    fn format_with_locale(
        &self,
        spec: &str,
        locale: &dyn RlCheckpointLocaleProvider,
    ) -> Result<String, String> {
        TrainingMetricScalar::Float(*self).format_with_locale(spec, locale)
    }
    fn representation(&self, repr: bool) -> Result<String, String> {
        TrainingMetricScalar::Float(*self).representation(repr)
    }
    fn attribute(&self, name: &str) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        TrainingMetricScalar::Float(*self).attribute(name)
    }
    fn item(
        &self,
        key: RlCheckpointFieldIndex<'_>,
    ) -> Result<Arc<dyn RlCheckpointFormatValue>, String> {
        TrainingMetricScalar::Float(*self).item(key)
    }
}

enum Value<'a> {
    Borrowed(&'a dyn RlCheckpointFormatValue),
    Owned(Arc<dyn RlCheckpointFormatValue>),
}
impl Value<'_> {
    fn get(&self) -> &dyn RlCheckpointFormatValue {
        match self {
            Self::Borrowed(value) => *value,
            Self::Owned(value) => value.as_ref(),
        }
    }
}

pub(crate) fn ascii_repr(text: &str) -> String {
    let mut result = String::new();
    for ch in text.chars() {
        if ch.is_ascii() {
            result.push(ch);
        } else {
            let point = u32::from(ch);
            // Python ascii() escapes only non-ASCII in the repr result; quoting/backslashes
            // already produced by repr must stay untouched. Rust escape_default differs.
            let rendered = match point {
                0..=0xff => write!(result, "\\x{point:02x}"),
                0x100..=0xffff => write!(result, "\\u{point:04x}"),
                _ => write!(result, "\\U{point:08x}"),
            };
            rendered.expect("writing to String cannot fail");
        }
    }
    result
}

/// Default keyword-only Checkpoint filename adapter. Formatting is incremental, preserving
/// earlier custom effects on a later error. Primitive values use the scalar layer's C locale
/// and Unicode-scalar boundaries; model objects implement explicit protocols above.
#[derive(Default)]
pub struct PythonRlCheckpointName;

impl<M: RlCheckpointFormatValue> RlCheckpointName<M> for PythonRlCheckpointName {
    fn render(
        &mut self,
        template: &str,
        iteration: &BigInt,
        local_time: &str,
        metrics: &IndexMap<String, M>,
    ) -> Result<String, String> {
        render_keywords(
            template,
            iteration,
            local_time,
            metrics,
            &RlCheckpointNumericLocale::default(),
        )
    }
}

/// Checkpoint names with a per-field numeric locale provider. The caller supplies
/// the actual runtime's locale; this does not equate OS defaults with `LC_NUMERIC`.
pub struct LocalizedPythonRlCheckpointName {
    locale: Arc<dyn RlCheckpointLocaleProvider>,
}

impl LocalizedPythonRlCheckpointName {
    #[must_use]
    pub fn new(locale: Arc<dyn RlCheckpointLocaleProvider>) -> Self {
        Self { locale }
    }
}

impl<M: RlCheckpointFormatValue> RlCheckpointName<M> for LocalizedPythonRlCheckpointName {
    fn render(
        &mut self,
        template: &str,
        iteration: &BigInt,
        local_time: &str,
        metrics: &IndexMap<String, M>,
    ) -> Result<String, String> {
        render_keywords(
            template,
            iteration,
            local_time,
            metrics,
            self.locale.as_ref(),
        )
    }
}

fn render_keywords<M: RlCheckpointFormatValue>(
    template: &str,
    iteration: &BigInt,
    local_time: &str,
    metrics: &IndexMap<String, M>,
    locale: &dyn RlCheckpointLocaleProvider,
) -> Result<String, String> {
    for key in ["iter", "time"] {
        if metrics.contains_key(key) {
            return Err(format!("duplicate checkpoint keyword {key}"));
        }
    }
    let iteration = TrainingMetricScalar::Integer(iteration.clone());
    let time = TrainingMetricScalar::Text(local_time.into());
    render(
        template,
        2,
        &|name| match name {
            "iter" => Ok(&iteration as &dyn RlCheckpointFormatValue),
            "time" => Ok(&time as &dyn RlCheckpointFormatValue),
            _ => metrics
                .get(name)
                .map(|value| value as &dyn RlCheckpointFormatValue)
                .ok_or_else(|| format!("missing checkpoint keyword {name}")),
        },
        locale,
    )
}

fn resolve<'a>(
    name: &str,
    lookup: &impl Fn(&str) -> Result<&'a dyn RlCheckpointFormatValue, String>,
) -> Result<Value<'a>, String> {
    let end = name.find(['.', '[']).unwrap_or(name.len());
    let first = &name[..end];
    if first.is_empty() || integer_index(first)?.is_some() {
        return Err("checkpoint format has no positional arguments".into());
    }
    let mut value = Value::Borrowed(lookup(first)?);
    let mut rest = &name[end..];
    while !rest.is_empty() {
        let (attribute, key) = part(&mut rest)?;
        let next = if attribute {
            value.get().attribute(key)
        } else {
            let index = integer_index(key)?.map_or(
                RlCheckpointFieldIndex::Key(key),
                RlCheckpointFieldIndex::Integer,
            );
            value.get().item(index)
        }?;
        value = Value::Owned(next);
    }
    Ok(value)
}

fn render<'a>(
    mut template: &str,
    depth: u8,
    lookup: &impl Fn(&str) -> Result<&'a dyn RlCheckpointFormatValue, String>,
    locale: &dyn RlCheckpointLocaleProvider,
) -> Result<String, String> {
    if depth == 0 {
        return Err("checkpoint format recursion exceeded".into());
    }
    let mut output = String::new();
    while let Some(index) = template.find(['{', '}']) {
        output.push_str(&template[..index]);
        let brace = template.as_bytes()[index];
        template = &template[index + 1..];
        if template.as_bytes().first() == Some(&brace) {
            output.push(char::from(brace));
            template = &template[1..];
            continue;
        }
        if brace == b'}' {
            return Err("single '}' in checkpoint format".into());
        }
        let field = field(&mut template)?;
        let mut value = resolve(field.name, lookup)?;
        if let Some(conversion) = field.conversion.filter(|ch| *ch != '\0') {
            let text = match conversion {
                's' => value.get().representation(false)?,
                'r' => value.get().representation(true)?,
                'a' => ascii_repr(&value.get().representation(true)?),
                _ => return Err(format!("unknown checkpoint conversion {conversion}")),
            };
            value = Value::Owned(Arc::new(TrainingMetricScalar::Text(text)));
        }
        let spec = if field.spec.contains('{') {
            render(field.spec, depth - 1, lookup, locale)?
        } else {
            field.spec.into()
        };
        output.push_str(&value.get().format_with_locale(&spec, locale)?);
    }
    output.push_str(template);
    Ok(output)
}

#[cfg(test)]
#[path = "../tests/support/rl_checkpoint_name.rs"]
mod tests;

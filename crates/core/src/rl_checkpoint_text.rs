//! Lossless Python-string checkpoint names backed by validated Unicode code points.

use std::{
    ffi::OsString,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use indexmap::IndexMap;
use num_bigint::BigInt;
use widestring::U32String;

use crate::rl_checkpoint_name::{ascii_repr, python_code_points_repr};
use crate::rl_checkpoint_numeric::decimal_digit;
use crate::{RlCheckpointLocaleProvider, RlCheckpointNumericLocale};

/// An owned Python string. Unlike Rust `String`, this includes lone surrogate code points.
/// The third-party storage type is deliberately private so plugin contracts remain stable.
#[derive(Clone, Default, Eq, Hash, PartialEq)]
pub struct RlCheckpointText(U32String);

impl RlCheckpointText {
    /// Validates Python's U+0000..=U+10FFFF code-point domain.
    /// # Errors
    /// Returns the first value outside Python's string range.
    pub fn try_from_code_points(points: impl IntoIterator<Item = u32>) -> Result<Self, String> {
        Self::checked(points.into_iter().collect())
    }

    fn checked(points: Vec<u32>) -> Result<Self, String> {
        if let Some(point) = points.iter().find(|point| **point > 0x10_ffff) {
            return Err(format!("checkpoint code point U+{point:X} is out of range"));
        }
        Ok(Self(U32String::from_vec(points)))
    }

    #[must_use]
    pub fn from_utf8(text: &str) -> Self {
        Self(U32String::from_str(text))
    }

    #[must_use]
    pub fn as_code_points(&self) -> &[u32] {
        self.0.as_slice()
    }

    // Slicing a validated text preserves its code-point invariant. Internal
    // parsers take this type, not arbitrary u32 slices, so no revalidation is needed.
    fn sliced(&self, range: std::ops::Range<usize>) -> Self {
        Self(U32String::from_vec(self.as_code_points()[range].to_vec()))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Converts only scalar-only text to UTF-8.
    /// # Errors
    /// Lone surrogates cannot be represented by `String`.
    pub fn to_utf8(&self) -> Result<String, String> {
        self.0.to_string().map_err(|error| error.to_string())
    }

    /// Converts to an OS path component without replacement or normalization.
    /// # Errors
    /// Non-Windows platforms reject lone surrogates because this contract is Windows UTF-16.
    pub fn to_os_string(&self) -> Result<OsString, String> {
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            Ok(OsString::from_wide(&self.windows_units()))
        }
        #[cfg(not(windows))]
        {
            self.to_utf8().map(OsString::from)
        }
    }

    /// Joins this raw name to an existing OS-native directory.
    /// # Errors
    /// Returns the platform conversion failure described by `to_os_string`.
    pub fn join_to(&self, directory: &Path) -> Result<PathBuf, String> {
        self.to_os_string().map(|name| directory.join(name))
    }

    #[cfg(windows)]
    fn windows_units(&self) -> Vec<u16> {
        let mut units = Vec::new();
        for &point in self.as_code_points() {
            if let Some(ch) = char::from_u32(point) {
                let mut buffer = [0; 2];
                units.extend_from_slice(ch.encode_utf16(&mut buffer));
            } else {
                // Validation admits no non-scalar values other than the surrogate range.
                units.push(u16::try_from(point).expect("surrogate code points fit in u16"));
            }
        }
        units
    }
}

impl From<&str> for RlCheckpointText {
    fn from(value: &str) -> Self {
        Self::from_utf8(value)
    }
}
impl From<String> for RlCheckpointText {
    fn from(value: String) -> Self {
        Self::from_utf8(&value)
    }
}
impl fmt::Debug for RlCheckpointText {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_list()
            .entries(self.as_code_points())
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RlLosslessCheckpointFieldIndex {
    Integer(usize),
    Key(RlCheckpointText),
}

pub trait RlLosslessCheckpointFormatValue: Send + Sync {
    /// # Errors
    /// Returns a value or format failure at the original field.
    fn format(&self, spec: &RlCheckpointText) -> Result<RlCheckpointText, String>;
    /// # Errors
    /// Returns a value, format, or locale-provider failure.
    fn format_with_locale(
        &self,
        spec: &RlCheckpointText,
        _locale: &dyn RlCheckpointLocaleProvider,
    ) -> Result<RlCheckpointText, String> {
        self.format(spec)
    }
    /// # Errors
    /// Returns a model conversion failure.
    fn representation(&self, repr: bool) -> Result<RlCheckpointText, String>;
    /// # Errors
    /// Returns a model attribute lookup failure.
    fn attribute(
        &self,
        name: &RlCheckpointText,
    ) -> Result<Arc<dyn RlLosslessCheckpointFormatValue>, String>;
    /// # Errors
    /// Returns a model item lookup failure.
    fn item(
        &self,
        key: RlLosslessCheckpointFieldIndex,
    ) -> Result<Arc<dyn RlLosslessCheckpointFormatValue>, String>;
}

impl<T: RlLosslessCheckpointFormatValue + ?Sized> RlLosslessCheckpointFormatValue for Arc<T> {
    fn format(&self, spec: &RlCheckpointText) -> Result<RlCheckpointText, String> {
        self.as_ref().format(spec)
    }
    fn format_with_locale(
        &self,
        spec: &RlCheckpointText,
        locale: &dyn RlCheckpointLocaleProvider,
    ) -> Result<RlCheckpointText, String> {
        self.as_ref().format_with_locale(spec, locale)
    }
    fn representation(&self, repr: bool) -> Result<RlCheckpointText, String> {
        self.as_ref().representation(repr)
    }
    fn attribute(
        &self,
        name: &RlCheckpointText,
    ) -> Result<Arc<dyn RlLosslessCheckpointFormatValue>, String> {
        self.as_ref().attribute(name)
    }
    fn item(
        &self,
        key: RlLosslessCheckpointFieldIndex,
    ) -> Result<Arc<dyn RlLosslessCheckpointFormatValue>, String> {
        self.as_ref().item(key)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RlLosslessCheckpointValue {
    Integer(BigInt),
    Float(f64),
    Boolean(bool),
    Text(RlCheckpointText),
    None,
}

impl RlLosslessCheckpointFormatValue for RlLosslessCheckpointValue {
    fn format(&self, spec: &RlCheckpointText) -> Result<RlCheckpointText, String> {
        self.format_with_locale(spec, &RlCheckpointNumericLocale::default())
    }
    fn format_with_locale(
        &self,
        spec: &RlCheckpointText,
        locale: &dyn RlCheckpointLocaleProvider,
    ) -> Result<RlCheckpointText, String> {
        crate::rl_checkpoint_text_format::format(self, spec, locale)
    }
    fn representation(&self, repr: bool) -> Result<RlCheckpointText, String> {
        if let Self::Text(text) = self {
            if !repr {
                return Ok(text.clone());
            }
            return Ok(python_code_points_repr(text.as_code_points().iter().copied()).into());
        }
        self.format(&RlCheckpointText::default())
    }
    fn attribute(
        &self,
        name: &RlCheckpointText,
    ) -> Result<Arc<dyn RlLosslessCheckpointFormatValue>, String> {
        let name = name.to_utf8()?;
        let result = match (self, name.as_str()) {
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
        key: RlLosslessCheckpointFieldIndex,
    ) -> Result<Arc<dyn RlLosslessCheckpointFormatValue>, String> {
        if let (Self::Text(text), RlLosslessCheckpointFieldIndex::Integer(index)) = (self, &key) {
            text.as_code_points()
                .get(*index)
                .ok_or("checkpoint string index out of range")?;
            return Ok(Arc::new(Self::Text(text.sliced(*index..*index + 1))));
        }
        Err(format!("unsupported checkpoint item {key:?}"))
    }
}

pub trait RlLosslessCheckpointName<M> {
    /// # Errors
    /// Returns malformed template, lookup, conversion, or value formatting failures.
    fn render(
        &mut self,
        template: &RlCheckpointText,
        iteration: &BigInt,
        local_time: &RlCheckpointText,
        metrics: &IndexMap<RlCheckpointText, M>,
    ) -> Result<RlCheckpointText, String>;
}

#[derive(Default)]
pub struct PythonRlLosslessCheckpointName;
impl<M: RlLosslessCheckpointFormatValue> RlLosslessCheckpointName<M>
    for PythonRlLosslessCheckpointName
{
    fn render(
        &mut self,
        template: &RlCheckpointText,
        iteration: &BigInt,
        local_time: &RlCheckpointText,
        metrics: &IndexMap<RlCheckpointText, M>,
    ) -> Result<RlCheckpointText, String> {
        render_keywords(
            template,
            iteration,
            local_time,
            metrics,
            &RlCheckpointNumericLocale::default(),
        )
    }
}

pub struct LocalizedPythonRlLosslessCheckpointName {
    locale: Arc<dyn RlCheckpointLocaleProvider>,
}
impl LocalizedPythonRlLosslessCheckpointName {
    #[must_use]
    pub fn new(locale: Arc<dyn RlCheckpointLocaleProvider>) -> Self {
        Self { locale }
    }
}
impl<M: RlLosslessCheckpointFormatValue> RlLosslessCheckpointName<M>
    for LocalizedPythonRlLosslessCheckpointName
{
    fn render(
        &mut self,
        template: &RlCheckpointText,
        iteration: &BigInt,
        local_time: &RlCheckpointText,
        metrics: &IndexMap<RlCheckpointText, M>,
    ) -> Result<RlCheckpointText, String> {
        render_keywords(
            template,
            iteration,
            local_time,
            metrics,
            self.locale.as_ref(),
        )
    }
}

struct Field {
    name: RlCheckpointText,
    conversion: Option<u32>,
    spec: RlCheckpointText,
}
fn parse_field(text: &RlCheckpointText, start: usize) -> Result<(Field, usize), String> {
    let input = text.as_code_points();
    let mut index = start;
    let mut bracket = false;
    let (name_end, delimiter) = loop {
        let point = *input.get(index).ok_or("unclosed checkpoint field")?;
        if bracket {
            if point == u32::from(']') {
                bracket = false;
            }
            index += 1;
            continue;
        }
        match char::from_u32(point) {
            Some('[') => bracket = true,
            Some('}' | ':' | '!') => break (index, point),
            Some('{') => return Err("unexpected '{' in checkpoint field".into()),
            _ => {}
        }
        index += 1;
    };
    // The loop can break only outside brackets; an unclosed bracket reaches
    // the input-exhaustion error inside the loop instead.
    let name = text.sliced(start..name_end);
    index += 1;
    if delimiter == u32::from('}') {
        return Ok((
            Field {
                name,
                conversion: None,
                spec: RlCheckpointText::default(),
            },
            index,
        ));
    }
    let mut conversion = None;
    if delimiter == u32::from('!') {
        conversion = Some(*input.get(index).ok_or("missing checkpoint conversion")?);
        index += 1;
        match input.get(index).copied() {
            Some(point) if point == u32::from('}') => {
                return Ok((
                    Field {
                        name,
                        conversion,
                        spec: RlCheckpointText::default(),
                    },
                    index + 1,
                ));
            }
            Some(point) if point == u32::from(':') => index += 1,
            _ => return Err("expected ':' after checkpoint conversion".into()),
        }
    }
    let spec_start = index;
    let mut depth = 1_usize;
    while let Some(point) = input.get(index).copied() {
        if point == u32::from('{') {
            depth += 1;
        }
        if point == u32::from('}') {
            depth -= 1;
            if depth == 0 {
                return Ok((
                    Field {
                        name,
                        conversion,
                        spec: text.sliced(spec_start..index),
                    },
                    index + 1,
                ));
            }
        }
        index += 1;
    }
    Err("unclosed checkpoint format specification".into())
}

fn integer_index(text: &RlCheckpointText) -> Result<Option<usize>, String> {
    if text.is_empty() {
        return Ok(None);
    }
    let mut value = 0_usize;
    for &point in text.as_code_points() {
        let Some(ch) = char::from_u32(point) else {
            return Ok(None);
        };
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

enum Value<'a> {
    Borrowed(&'a dyn RlLosslessCheckpointFormatValue),
    Owned(Arc<dyn RlLosslessCheckpointFormatValue>),
}
impl Value<'_> {
    fn get(&self) -> &dyn RlLosslessCheckpointFormatValue {
        match self {
            Self::Borrowed(value) => *value,
            Self::Owned(value) => value.as_ref(),
        }
    }
}

fn resolve<'a>(
    name: &RlCheckpointText,
    lookup: &dyn Fn(&RlCheckpointText) -> Result<&'a dyn RlLosslessCheckpointFormatValue, String>,
) -> Result<Value<'a>, String> {
    let points = name.as_code_points();
    let end = points
        .iter()
        .position(|point| *point == u32::from('.') || *point == u32::from('['))
        .unwrap_or(points.len());
    let first = name.sliced(0..end);
    if first.is_empty() || integer_index(&first)?.is_some() {
        return Err("checkpoint format has no positional arguments".into());
    }
    let mut value = Value::Borrowed(lookup(&first)?);
    let mut index = end;
    while index < points.len() {
        let attribute = points[index] == u32::from('.');
        if !attribute && points[index] != u32::from('[') {
            return Err("only '.' or '[' may follow a checkpoint item".into());
        }
        index += 1;
        let start = index;
        if attribute {
            while index < points.len()
                && points[index] != u32::from('.')
                && points[index] != u32::from('[')
            {
                index += 1;
            }
        } else {
            while index < points.len() && points[index] != u32::from(']') {
                index += 1;
            }
            if index == points.len() {
                return Err("missing ']' in checkpoint field".into());
            }
        }
        if start == index {
            return Err("empty checkpoint attribute or index".into());
        }
        let part = name.sliced(start..index);
        if !attribute {
            index += 1;
        }
        value = Value::Owned(if attribute {
            value.get().attribute(&part)?
        } else {
            value.get().item(integer_index(&part)?.map_or(
                RlLosslessCheckpointFieldIndex::Key(part),
                RlLosslessCheckpointFieldIndex::Integer,
            ))?
        });
    }
    Ok(value)
}

fn ascii_lossless(text: &RlCheckpointText) -> RlCheckpointText {
    let mut output = Vec::new();
    for &point in text.as_code_points() {
        if point <= 0x7f {
            output.push(point);
        } else {
            output.extend(
                ascii_repr(
                    &char::from_u32(point)
                        .map_or_else(|| format!("\\u{point:04x}"), |ch| ch.to_string()),
                )
                .chars()
                .map(u32::from),
            );
        }
    }
    RlCheckpointText::try_from_code_points(output).expect("ASCII escapes are valid")
}

fn render<'a>(
    template: &RlCheckpointText,
    depth: u8,
    lookup: &dyn Fn(&RlCheckpointText) -> Result<&'a dyn RlLosslessCheckpointFormatValue, String>,
    locale: &dyn RlCheckpointLocaleProvider,
) -> Result<RlCheckpointText, String> {
    if depth == 0 {
        return Err("checkpoint format recursion exceeded".into());
    }
    let points = template.as_code_points();
    let mut output = Vec::new();
    let mut index = 0;
    while index < points.len() {
        if points[index] != u32::from('{') && points[index] != u32::from('}') {
            output.push(points[index]);
            index += 1;
            continue;
        }
        let brace = points[index];
        if points.get(index + 1) == Some(&brace) {
            output.push(brace);
            index += 2;
            continue;
        }
        if brace == u32::from('}') {
            return Err("single '}' in checkpoint format".into());
        }
        let (field, next) = parse_field(template, index + 1)?;
        index = next;
        let mut value = resolve(&field.name, lookup)?;
        if let Some(conversion) = field.conversion.filter(|point| *point != 0) {
            let text = match char::from_u32(conversion) {
                Some('s') => value.get().representation(false)?,
                Some('r') => value.get().representation(true)?,
                Some('a') => ascii_lossless(&value.get().representation(true)?),
                _ => return Err(format!("unknown checkpoint conversion U+{conversion:04X}")),
            };
            value = Value::Owned(Arc::new(RlLosslessCheckpointValue::Text(text)));
        }
        let spec = if field.spec.as_code_points().contains(&u32::from('{')) {
            render(&field.spec, depth - 1, lookup, locale)?
        } else {
            field.spec
        };
        output.extend_from_slice(
            value
                .get()
                .format_with_locale(&spec, locale)?
                .as_code_points(),
        );
    }
    // Every fragment came from validated text or a value returning that type.
    Ok(RlCheckpointText(U32String::from_vec(output)))
}

// Erase only the metric table's value type at the parser boundary. Public
// callers retain typed IndexMaps, while every value protocol uses one parser
// and the same validation/failure order rather than duplicated monomorphs.
trait Keywords {
    fn value(&self, name: &RlCheckpointText) -> Option<&dyn RlLosslessCheckpointFormatValue>;
}
impl<M: RlLosslessCheckpointFormatValue> Keywords for IndexMap<RlCheckpointText, M> {
    fn value(&self, name: &RlCheckpointText) -> Option<&dyn RlLosslessCheckpointFormatValue> {
        self.get(name)
            .map(|value| value as &dyn RlLosslessCheckpointFormatValue)
    }
}

fn render_keywords(
    template: &RlCheckpointText,
    iteration: &BigInt,
    local_time: &RlCheckpointText,
    metrics: &dyn Keywords,
    locale: &dyn RlCheckpointLocaleProvider,
) -> Result<RlCheckpointText, String> {
    for key in ["iter", "time"] {
        if metrics.value(&RlCheckpointText::from(key)).is_some() {
            return Err(format!("duplicate checkpoint keyword {key}"));
        }
    }
    let iteration = RlLosslessCheckpointValue::Integer(iteration.clone());
    let time = RlLosslessCheckpointValue::Text(local_time.clone());
    render(
        template,
        2,
        &|name| {
            if name == &RlCheckpointText::from("iter") {
                Ok(&iteration as &dyn RlLosslessCheckpointFormatValue)
            } else if name == &RlCheckpointText::from("time") {
                Ok(&time as &dyn RlLosslessCheckpointFormatValue)
            } else {
                metrics
                    .value(name)
                    .ok_or_else(|| format!("missing checkpoint keyword {name:?}"))
            }
        },
        locale,
    )
}

#[cfg(test)]
#[path = "../tests/support/rl_checkpoint_text.rs"]
mod tests;

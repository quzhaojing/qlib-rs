//! Unicode numeric syntax shared by checkpoint field lookup and scalar formats.

use std::borrow::Cow;

pub(crate) fn decimal_digit(ch: char) -> Option<usize> {
    use intl::unicode::{NumericType, age, numeric_type, numeric_value};
    // Match CPython 3.14's Unicode 16, not newly assigned Unicode 17 digits.
    if numeric_type(ch) == Some(NumericType::Decimal)
        && age(ch).expect("decimal digits have an assignment age") <= (16, 0)
    {
        Some(
            usize::try_from(
                numeric_value(ch)
                    .expect("decimal digits have a numeric value")
                    .numerator,
            )
            .expect("decimal digit is in 0..=9"),
        )
    } else {
        None
    }
}

fn number(remaining: &mut &str, output: &mut String) -> Result<(), String> {
    let mut value = 0_usize;
    let mut end = 0;
    for (offset, ch) in remaining.char_indices() {
        let Some(digit) = decimal_digit(ch) else {
            break;
        };
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add(digit))
            .filter(|value| isize::try_from(*value).is_ok())
            .ok_or("format numeric field is too large")?;
        end = offset + ch.len_utf8();
    }
    if end != 0 {
        // Strip leading numeric zeroes: Unicode zero must not become a zero-fill
        // flag. Keep an all-zero width nonempty for Boolean/None format behavior.
        output.push_str(&value.to_string());
        *remaining = &remaining[end..];
    }
    Ok(())
}

pub(crate) fn normalize_numeric_fields(spec: &str) -> Result<Cow<'_, str>, String> {
    if spec.is_ascii() {
        return Ok(Cow::Borrowed(spec));
    }
    let mut chars = spec.char_indices();
    let first = chars.next();
    let second = chars.next();
    let prefix_end = match (first, second) {
        (_, Some((offset, '<' | '>' | '=' | '^'))) => offset + 1,
        (Some((_, '<' | '>' | '=' | '^')), _) => 1,
        _ => 0,
    };
    let mut output = spec[..prefix_end].to_owned();
    let mut remaining = &spec[prefix_end..];
    // Preserve flags verbatim, especially ASCII 0, and never normalize a fill.
    for options in ["+- ", "z", "#", "0"] {
        if let Some(ch) = remaining.chars().next().filter(|ch| options.contains(*ch)) {
            output.push(ch);
            remaining = &remaining[ch.len_utf8()..];
        }
    }
    number(&mut remaining, &mut output)?;
    if let Some(ch @ (',' | '_')) = remaining.chars().next() {
        output.push(ch);
        remaining = &remaining[1..];
    }
    if let Some(rest) = remaining.strip_prefix('.') {
        output.push('.');
        remaining = rest;
        number(&mut remaining, &mut output)?;
    }
    // Leave type/invalid suffix/fractional-grouping syntax to the format engine.
    output.push_str(remaining);
    Ok(Cow::Owned(output))
}

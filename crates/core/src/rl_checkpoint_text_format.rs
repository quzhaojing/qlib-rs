//! Positional lossless-text adapters around the existing scalar format engines.
//! Tags need not be absent from user text or locale metadata.
use crate::{
    RlCheckpointLocaleProvider, RlCheckpointNumericLocale, RlCheckpointText,
    RlLosslessCheckpointValue, TrainingMetricScalar, format_rl_checkpoint_scalar_with_locale,
};
use num_traits::ToPrimitive;
use std::sync::OnceLock;

const FILL_A: char = '\u{e000}';
const FILL_B: char = '\u{e001}';
const CHARACTER_A: char = '\u{e002}';
const CHARACTER_B: char = '\u{e003}';

fn tagged_spec(spec: &RlCheckpointText, marker: char) -> String {
    spec.as_code_points()
        .iter()
        .map(|&point| char::from_u32(point).unwrap_or(marker))
        .collect()
}

fn text_format(
    text: &RlCheckpointText,
    spec: &RlCheckpointText,
    locale: &dyn RlCheckpointLocaleProvider,
) -> Result<RlCheckpointText, String> {
    // The pinned string engine only truncates code points and pads. A sentinel
    // distinct from the fill lets it compute layout without receiving user data.
    let sentinel = if spec.as_code_points().first() == Some(&u32::from('x')) {
        'y'
    } else {
        'x'
    };
    let dummy = TrainingMetricScalar::Text(sentinel.to_string().repeat(text.len()));
    let layout =
        format_rl_checkpoint_scalar_with_locale(&dummy, &tagged_spec(spec, FILL_A), locale)?;
    let raw_fill = spec
        .as_code_points()
        .first()
        .copied()
        .filter(|point| char::from_u32(*point).is_none());
    let mut content = text.as_code_points().iter();
    let points = layout.chars().map(|ch| {
        if ch == sentinel {
            *content
                .next()
                .expect("string formatting only truncates the original content")
        } else if ch == FILL_A {
            raw_fill.unwrap_or(u32::from(ch))
        } else {
            u32::from(ch)
        }
    });
    Ok(RlCheckpointText::try_from_code_points(points)
        .expect("original text and fill are validated code points"))
}

struct CapturedLocale<'a> {
    source: &'a dyn RlCheckpointLocaleProvider,
    snapshot: OnceLock<RlCheckpointNumericLocale>,
}
impl RlCheckpointLocaleProvider for CapturedLocale<'_> {
    fn numeric_locale(&self) -> Result<RlCheckpointNumericLocale, String> {
        let snapshot = self.source.numeric_locale()?;
        let _ = self.snapshot.set(snapshot.clone());
        Ok(snapshot)
    }
}

pub(crate) fn format(
    value: &RlLosslessCheckpointValue,
    spec: &RlCheckpointText,
    locale: &dyn RlCheckpointLocaleProvider,
) -> Result<RlCheckpointText, String> {
    let (scalar, raw_character) = match value {
        RlLosslessCheckpointValue::Text(text) => return text_format(text, spec, locale),
        RlLosslessCheckpointValue::Integer(value) => {
            let point = if spec.as_code_points().last() == Some(&u32::from('c')) {
                value
                    .to_u32()
                    .filter(|point| (0xd800..=0xdfff).contains(point))
            } else {
                None
            };
            (TrainingMetricScalar::Integer(value.clone()), point)
        }
        RlLosslessCheckpointValue::Float(value) => (TrainingMetricScalar::Float(*value), None),
        RlLosslessCheckpointValue::Boolean(value) => (TrainingMetricScalar::Boolean(*value), None),
        RlLosslessCheckpointValue::None => (TrainingMetricScalar::Null, None),
    };
    let has_surrogates = spec
        .as_code_points()
        .iter()
        .any(|&point| char::from_u32(point).is_none());
    if !has_surrogates && raw_character.is_none() {
        return format_rl_checkpoint_scalar_with_locale(
            &scalar,
            &tagged_spec(spec, FILL_A),
            locale,
        )
        .map(RlCheckpointText::from);
    }
    // Only padding and a surrogate `c` result change between the two tag sets.
    // Literal locale data stay equal at each position, even if they contain tags.
    // Query the real locale only in the first pass, after normal validation.
    let capture = CapturedLocale {
        source: locale,
        snapshot: OnceLock::new(),
    };
    let tagged_scalar = |marker: char| {
        if raw_character.is_some() {
            TrainingMetricScalar::Integer(u32::from(marker).into())
        } else {
            scalar.clone()
        }
    };
    let first = format_rl_checkpoint_scalar_with_locale(
        &tagged_scalar(CHARACTER_A),
        &tagged_spec(spec, FILL_A),
        &capture,
    )?;
    let snapshot = capture.snapshot.into_inner().unwrap_or_default();
    let second = format_rl_checkpoint_scalar_with_locale(
        &tagged_scalar(CHARACTER_B),
        &tagged_spec(spec, FILL_B),
        &snapshot,
    )
    .expect("changing one-code-point tags preserves validated numeric formatting");
    let points = first.chars().zip(second.chars()).map(|(first, second)| {
        if first == second {
            u32::from(first)
        } else if first == FILL_A {
            spec.as_code_points()[0]
        } else {
            raw_character.expect("only a tagged c character can differ outside padding")
        }
    });
    Ok(RlCheckpointText::try_from_code_points(points)
        .expect("only validated surrogates replace scalar tags"))
}

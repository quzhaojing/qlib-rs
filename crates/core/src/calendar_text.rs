//! Strict byte decoding for calendar file backends, without BOM sniffing or replacement.

use encoding_rs::GBK;
use thiserror::Error;

use crate::CalendarLoadError;

/// Explicit encoding policy. Host-default locale selection belongs to file opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalendarTextEncoding {
    Utf8,
    /// Python's strict GBK/CP936 codec, not the web-standard GB18030 superset.
    PythonGbk,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{encoding:?} decoding failed at bytes {start}..{end}: {reason}")]
pub struct CalendarTextDecodeError {
    pub encoding: CalendarTextEncoding,
    pub start: usize,
    pub end: usize,
    pub reason: &'static str,
}

impl From<CalendarTextDecodeError> for CalendarLoadError {
    fn from(error: CalendarTextDecodeError) -> Self {
        Self::Value(error.to_string())
    }
}

pub trait CalendarTextDecoder: Send + Sync {
    /// Decode a complete file buffer. Decoding errors must retain Value classification.
    ///
    /// # Errors
    /// Returns malformed input or decoder plugin failures.
    fn decode_text(&self, bytes: &[u8]) -> Result<String, CalendarLoadError>;
}

impl CalendarTextDecoder for CalendarTextEncoding {
    fn decode_text(&self, bytes: &[u8]) -> Result<String, CalendarLoadError> {
        self.decode(bytes).map_err(Into::into)
    }
}

impl CalendarTextEncoding {
    /// # Errors
    /// Returns the first malformed sequence with byte offsets. No partial text escapes.
    pub fn decode(self, bytes: &[u8]) -> Result<String, CalendarTextDecodeError> {
        match self {
            Self::Utf8 => std::str::from_utf8(bytes)
                .map(str::to_owned)
                .map_err(|error| CalendarTextDecodeError {
                    encoding: self,
                    start: error.valid_up_to(),
                    end: error
                        .error_len()
                        .map_or(bytes.len(), |length| error.valid_up_to() + length),
                    reason: if error.error_len().is_some() {
                        "invalid utf-8 sequence"
                    } else {
                        "incomplete utf-8 sequence"
                    },
                }),
            Self::PythonGbk => decode_gbk(bytes),
        }
    }
}

fn decode_gbk(bytes: &[u8]) -> Result<String, CalendarTextDecodeError> {
    let mut text = String::with_capacity(bytes.len());
    let mut index = 0;
    while let Some(&first) = bytes.get(index) {
        if first.is_ascii() {
            text.push(char::from(first));
            index += 1;
            continue;
        }
        let Some(&second) = bytes.get(index + 1) else {
            return Err(gbk_error(index, "incomplete multibyte sequence"));
        };
        let pair = [first, second];
        if first == 0x80 || is_gb18030_extension(u16::from_be_bytes(pair)) {
            return Err(gbk_error(index, "illegal multibyte sequence"));
        }
        let decoded = GBK
            .decode_without_bom_handling_and_without_replacement(&pair)
            .ok_or_else(|| gbk_error(index, "illegal multibyte sequence"))?;
        text.push_str(&decoded);
        index += 2;
    }
    Ok(text)
}

fn gbk_error(start: usize, reason: &'static str) -> CalendarTextDecodeError {
    CalendarTextDecodeError {
        encoding: CalendarTextEncoding::PythonGbk,
        start,
        end: start + 1,
        reason,
    }
}

// The pinned library maps every Python-valid GBK pair identically. These are its
// additional two-byte mappings, not a replacement character table. The exhaustive
// differential checks all 65,536 pairs on every test run; the probe records provenance.
fn is_gb18030_extension(pair: u16) -> bool {
    matches!(pair,
        0xa140..=0xa17e | 0xa180..=0xa1a0 | 0xa240..=0xa27e | 0xa280..=0xa2a0 |
        0xa2ab..=0xa2b0 | 0xa2e3..=0xa2e4 | 0xa2ef..=0xa2f0 | 0xa2fd..=0xa2fe |
        0xa340..=0xa37e | 0xa380..=0xa3a0 | 0xa440..=0xa47e | 0xa480..=0xa4a0 |
        0xa4f4..=0xa4fe | 0xa540..=0xa57e | 0xa580..=0xa5a0 | 0xa5f7..=0xa5fe |
        0xa640..=0xa67e | 0xa680..=0xa6a0 | 0xa6b9..=0xa6c0 | 0xa6d9..=0xa6df |
        0xa6ec..=0xa6ed | 0xa6f3 | 0xa6f6..=0xa6fe | 0xa740..=0xa77e |
        0xa780..=0xa7a0 | 0xa7c2..=0xa7d0 | 0xa7f2..=0xa7fe | 0xa896..=0xa8a0 |
        0xa8bc | 0xa8bf | 0xa8c1..=0xa8c4 | 0xa8ea..=0xa8fe | 0xa958 | 0xa95b |
        0xa95d..=0xa95f | 0xa989..=0xa995 | 0xa997..=0xa9a3 | 0xa9f0..=0xa9fe |
        0xaaa1..=0xaafe | 0xaba1..=0xabfe | 0xaca1..=0xacfe | 0xada1..=0xadfe |
        0xaea1..=0xaefe | 0xafa1..=0xaffe | 0xd7fa..=0xd7fe | 0xf8a1..=0xf8fe |
        0xf9a1..=0xf9fe | 0xfaa1..=0xfafe | 0xfba1..=0xfbfe | 0xfca1..=0xfcfe |
        0xfda1..=0xfdfe | 0xfe50..=0xfe7e | 0xfe80..=0xfefe)
}

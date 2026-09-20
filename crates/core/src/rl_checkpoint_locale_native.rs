//! Current-runtime locale provider for dynamically linked Windows MSVC hosts.

use crate::{RlCheckpointLocaleProvider, RlCheckpointNumericLocale};

/// Queries the current calling thread's UCRT locale at each primitive `n` field.
/// Use with `LocalizedPythonRlCheckpointName`; no Python runtime is embedded.
#[derive(Clone, Copy, Debug, Default)]
pub struct CurrentCrtRlCheckpointLocale;

impl RlCheckpointLocaleProvider for CurrentCrtRlCheckpointLocale {
    fn numeric_locale(&self) -> Result<RlCheckpointNumericLocale, String> {
        convert_snapshot(locale::current_numeric_locale())
    }
}

fn convert_snapshot(
    snapshot: Result<locale::NumericLocale, locale::LocaleError>,
) -> Result<RlCheckpointNumericLocale, String> {
    let snapshot = snapshot.map_err(|error| error.to_string())?;
    RlCheckpointNumericLocale::new(snapshot.decimal, snapshot.separator, &snapshot.grouping)
}

#[cfg(test)]
#[path = "../tests/support/rl_checkpoint_locale_native.rs"]
mod tests;

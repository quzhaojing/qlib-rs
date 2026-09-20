use super::*;
use crate::{LocalizedPythonRlCheckpointName, RlCheckpointName, TrainingMetricScalar};
use std::sync::Arc;

#[test]
fn current_runtime_provider_drives_real_filename_formatting() {
    let provider = CurrentCrtRlCheckpointLocale;
    // Rust test processes have C LC_NUMERIC. Locale mutation tests run in the
    // adapter crate's separate child process, never in this shared test process.
    assert_eq!(
        provider.numeric_locale().unwrap(),
        RlCheckpointNumericLocale::default()
    );
    let mut renderer = LocalizedPythonRlCheckpointName::new(Arc::new(provider));
    let metrics =
        indexmap::IndexMap::from([("value".to_string(), TrainingMetricScalar::Float(12345.6789))]);
    assert_eq!(
        renderer
            .render(
                "{iter:n}-{value:.10n}",
                &123_456_789.into(),
                "now",
                &metrics
            )
            .unwrap(),
        "123456789-12345.6789"
    );
}

#[test]
fn native_provider_preserves_acquisition_and_metadata_errors() {
    assert_eq!(
        convert_snapshot(Err(locale::LocaleError::Unavailable("acquire"))),
        Err("UCRT locale data unavailable at acquire".into())
    );
    assert_eq!(
        convert_snapshot(Ok(locale::NumericLocale {
            decimal: String::new(),
            separator: String::new(),
            grouping: vec![0],
        })),
        Err("locale decimal mark must not be empty".into())
    );
}

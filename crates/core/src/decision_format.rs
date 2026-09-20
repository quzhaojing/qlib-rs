//! Tree formatting for flat decisions collected from nested backtests.

/// Supplies the strategy-calendar frequency associated with one decision.
///
/// Implementations may query a live calendar. Calls are deliberately not cached
/// because the source function observes the calendar at every recursive level.
pub trait DecisionFrequency {
    /// Plugin-specific failure propagated without translation.
    type Error;

    /// Return the current Qlib frequency spelling.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error when the live calendar cannot supply
    /// its frequency.
    fn frequency(&self) -> Result<String, Self::Error>;
}

/// One decision and any lower-frequency decisions that follow it.
#[derive(Debug)]
pub struct FormattedDecisionItem<'a, D> {
    /// The original decision; formatting never clones it.
    pub decision: &'a D,
    /// Recursively formatted decisions before the next peer at this level.
    pub nested: Option<FormattedDecisions<'a, D>>,
}

/// A frequency level in the formatted decision tree.
#[derive(Debug)]
pub struct FormattedDecisions<'a, D> {
    /// Frequency returned by the first decision at this level.
    pub frequency: String,
    /// Stable source-ordered decisions at this frequency.
    pub items: Vec<FormattedDecisionItem<'a, D>>,
}

/// Reproduce `qlib.backtest.format_decisions` for a borrowed decision slice.
///
/// The first decision defines the current frequency. Every later decision is
/// queried in order; matching frequencies start a peer item, while intervening
/// decisions become that item's recursively formatted children.
///
/// # Errors
///
/// Returns the first frequency-provider error in the same observation order as
/// the source implementation.
pub fn format_decisions<D: DecisionFrequency>(
    decisions: &[D],
) -> Result<Option<FormattedDecisions<'_, D>>, D::Error> {
    let Some(first) = decisions.first() else {
        return Ok(None);
    };

    let frequency = first.frequency()?;
    let mut items = Vec::new();
    let mut last_decision_index = 0;

    for (index, decision) in decisions.iter().enumerate().skip(1) {
        if decision.frequency()? == frequency {
            items.push(FormattedDecisionItem {
                decision: &decisions[last_decision_index],
                nested: format_decisions(&decisions[last_decision_index + 1..index])?,
            });
            last_decision_index = index;
        }
    }

    items.push(FormattedDecisionItem {
        decision: &decisions[last_decision_index],
        nested: format_decisions(&decisions[last_decision_index + 1..])?,
    });
    Ok(Some(FormattedDecisions { frequency, items }))
}

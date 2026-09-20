//! Source-ordered representation of order-bearing trade decisions.

/// Dynamic values read by Python's `TradeDecisionWO.__repr__`.
///
/// Python evaluates these fields from left to right and stops at the first exception. Keeping the
/// operations behind one narrow context preserves that ordering without forcing a Rust decision to
/// own a Python strategy or expose Python object types in the domain model.
pub trait TradeDecisionReprContext {
    type Error;

    /// Runtime class name (`self.__class__.__name__`).
    ///
    /// # Errors
    ///
    /// Returns the adapter's class-metadata access error.
    fn class_name(&mut self) -> Result<String, Self::Error>;
    /// Python-format-compatible strategy text.
    ///
    /// # Errors
    ///
    /// Returns the adapter's strategy access or formatting error.
    fn strategy_text(&mut self) -> Result<String, Self::Error>;
    /// Python-format-compatible trade-range text, including `"None"` when absent.
    ///
    /// # Errors
    ///
    /// Returns the adapter's range access or formatting error.
    fn trade_range_text(&mut self) -> Result<String, Self::Error>;
    /// Current length of the live order list.
    ///
    /// # Errors
    ///
    /// Returns the adapter's order-list access or length error.
    fn order_count(&mut self) -> Result<usize, Self::Error>;
}

/// Format `TradeDecisionWO.__repr__` while preserving source field-access and failure order.
///
/// # Errors
///
/// Returns the first context error and does not evaluate any later field.
pub fn format_trade_decision_repr<C>(context: &mut C) -> Result<String, C::Error>
where
    C: TradeDecisionReprContext + ?Sized,
{
    let class_name = context.class_name()?;
    let strategy = context.strategy_text()?;
    let trade_range = context.trade_range_text()?;
    let order_count = context.order_count()?;

    Ok(format!(
        "class: {class_name}; strategy: {strategy}; trade_range: {trade_range}; order_list[{order_count}]"
    ))
}

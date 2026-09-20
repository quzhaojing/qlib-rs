//! Identity-preserving lookup for `qlib.rl.order_execution.utils`.

use thiserror::Error;

/// Borrowed executor topology, independent of execution and calendar ownership.
pub trait ExecutorLookup<Simulator: ?Sized, Failure> {
    /// Read the child of a nested executor; return `None` for other executor kinds.
    ///
    /// Adapters must test nested membership first, including types that also
    /// implement the simulator interface. Do not read a non-nested object's
    /// incidental `inner_executor` attribute.
    ///
    /// # Errors
    /// Returns the original child-property access failure.
    fn inner_executor(&self) -> Result<Option<&dyn ExecutorLookup<Simulator, Failure>>, Failure>;

    /// Borrow this object as a simulator, including simulator subclasses.
    fn as_simulator(&self) -> Option<&Simulator>;
}

/// Lookup failure preserves child-access errors and the source's empty assertion.
#[derive(Debug, PartialEq, Eq, Error)]
pub enum SimulatorLookupError<Failure> {
    #[error("{0}")]
    Access(Failure),
    #[error("")]
    NotSimulator,
}

/// Follow nested children and return the exact terminal simulator object.
///
/// Traversal is iterative, allocates nothing and does not clone executor state.
/// As in the source, cyclic nested graphs do not terminate.
///
/// # Errors
/// Returns a child-access failure immediately, or `NotSimulator` if the first
/// non-nested object is not a simulator.
pub fn get_simulator_executor<Simulator: ?Sized, Failure>(
    mut executor: &dyn ExecutorLookup<Simulator, Failure>,
) -> Result<&Simulator, SimulatorLookupError<Failure>> {
    while let Some(inner) = executor
        .inner_executor()
        .map_err(SimulatorLookupError::Access)?
    {
        executor = inner;
    }
    executor
        .as_simulator()
        .ok_or(SimulatorLookupError::NotSimulator)
}

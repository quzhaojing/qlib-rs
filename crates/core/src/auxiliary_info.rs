//! Generic auxiliary-information collection for reinforcement-learning environments.

/// Unconstrained auxiliary-information type.
pub type AuxInfoType<T> = T;

/// Collects customized auxiliary information from one simulator state.
pub trait AuxiliaryInfoCollector<State>: Send {
    type AuxiliaryInfo;
    type Error;

    /// Collect auxiliary information from `simulator_state`.
    ///
    /// # Errors
    /// Returns collector-specific failures.
    fn collect(&self, simulator_state: &State) -> Result<Self::AuxiliaryInfo, Self::Error>;

    /// Delegate to the concrete collection recipe.
    ///
    /// # Errors
    /// Returns collector-specific failures.
    fn call(&self, simulator_state: &State) -> Result<Self::AuxiliaryInfo, Self::Error> {
        self.collect(simulator_state)
    }
}

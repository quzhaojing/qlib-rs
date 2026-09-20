//! Generic reward contract shared by reinforcement-learning environments.

/// Computes a scalar reward from a simulator state.
///
/// `State` is deliberately unconstrained, matching Python's unconstrained
/// `SimulatorState` type variable. Implementations choose their own typed
/// failure domain. [`Self::call`] is the common invocation path and delegates
/// to [`Self::reward`], corresponding to the upstream final `__call__` method.
pub trait Reward<State>: Send + Sync {
    /// Failure returned by reward calculation.
    type Error;

    /// Implements the concrete reward recipe.
    ///
    /// # Errors
    /// Returns the implementation's native failure without wrapping it.
    fn reward(&self, simulator_state: &State) -> Result<f64, Self::Error>;

    /// Invokes the concrete reward recipe.
    ///
    /// # Errors
    /// Propagates [`Self::reward`] unchanged.
    fn call(&self, simulator_state: &State) -> Result<f64, Self::Error> {
        self.reward(simulator_state)
    }
}

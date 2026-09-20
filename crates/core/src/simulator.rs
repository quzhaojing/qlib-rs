//! Generic reinforcement-learning simulator contract.

/// Unconstrained simulator-state type.
pub type StateType<T> = T;

/// Unconstrained simulator-action type.
pub type ActType<T> = T;

/// Ephemeral simulator created from `InitialState`, exposing `State` and
/// consuming `Action` values.
///
/// Construction is deliberately left to a factory: the upstream base
/// constructor accepts an initial value and arbitrary keyword arguments but
/// stores nothing. Requiring the three transition methods at compile time is
/// the native counterpart of their default `NotImplementedError` bodies.
pub trait Simulator<InitialState, State, Action>: Send {
    /// Failure domain selected by the concrete simulator.
    type Error;

    /// Applies one action and updates internal state.
    ///
    /// # Errors
    /// Returns the concrete transition failure unchanged.
    fn step(&mut self, action: Action) -> Result<(), Self::Error>;

    /// Retrieves the current state.
    ///
    /// # Errors
    /// Returns the concrete state-read failure unchanged.
    fn get_state(&self) -> Result<State, Self::Error>;

    /// Reports whether the trajectory has ended.
    ///
    /// # Errors
    /// Returns the concrete terminal-state failure unchanged.
    fn done(&self) -> Result<bool, Self::Error>;
}

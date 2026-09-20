//! Initial-state type contract corresponding to `qlib.rl.seed`.

/// Unconstrained simulator initial-state type.
///
/// This is a zero-cost counterpart of Python's unconstrained
/// `InitialStateType` type variable. The concrete type remains exactly `T`.
pub type InitialStateType<T> = T;

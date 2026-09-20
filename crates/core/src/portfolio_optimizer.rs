//! Abstract portfolio-optimizer contract.

/// Constructs a portfolio using an optimization method.
///
/// `Arguments` deliberately remains unconstrained: the upstream abstract
/// `__call__` accepts arbitrary positional and keyword arguments, and its two
/// current implementations expose different input shapes. A tuple or request
/// record can carry those arguments without imposing a common runtime wrapper.
pub trait BaseOptimizer<Arguments> {
    /// Concrete allocation returned by this optimizer.
    type Output;

    /// Generates an optimized portfolio allocation.
    fn call(&self, arguments: Arguments) -> Self::Output;
}

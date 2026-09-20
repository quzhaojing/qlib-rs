// Keep both numeric families in one executable for combined LLVM measurement.
// Reuse the exact standalone test suites, including both live NumPy oracles.
#[path = "complex_price_advantage.rs"]
mod complex;
#[path = "price_advantage.rs"]
mod real;

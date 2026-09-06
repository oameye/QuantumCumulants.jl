# Structured moment-polynomial representation

`MomentIR` is the evaluator-independent numerical representation used to preserve the
polynomial structure of completed deterministic cumulant hierarchies.

This is currently developer-facing. It intentionally contains symbolic metadata used during
construction; a numerical evaluator must compile that metadata into concrete runtime storage
and must not inspect symbolic fields on its hot RHS path.

The representation has four invariants:

1. Row/state `i` is exactly `eqs.states[i]` / `eqs.equations[i]`. Lowering never derives
   numerical ordering independently from graph iteration.
2. Monomial `1` is the empty product. Every later monomial is prefix-closed and references a
   parent with a smaller id. Positive state factors denote stored states and negative factors
   denote the conjugate of a stored state.
3. Equation rows are direct contiguous term ranges (`rowptr`) over shared monomial ids and
   pooled symbolic coefficient ids. The IR does not prescribe a sparse-matrix implementation.
4. Unsupported state-dependent syntax is an explicit capability error. Generic Symbolics
   polynomial conversion is a validation oracle, not a hidden production fallback.

The first supported polynomial grammar is sums, products, unary/binary subtraction,
nonnegative integer powers, and division by state-independent expressions. Any state-free
subtree is retained as a symbolic coefficient.

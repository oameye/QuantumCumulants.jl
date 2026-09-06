# Benchmark-only prototype for lowering completed QuantumCumulants drifts directly to
# MomentIR without sending every equation through Symbolics.polynomial_coeffs.
#
# This file deliberately does NOT change the production backend. It tests two hypotheses:
#   1. ordinary completed QC drifts already have a sum-of-coefficient×moment-monomial shape
#      that can be extracted directly;
#   2. prefix interning by (parent_id, leaf) avoids Vector-key hashing and recursive slicing.
#
# Unsupported state-dependent expression shapes fall back, equation-by-equation, to the
# existing Symbolics.polynomial_coeffs path. The benchmark reports the fallback count.
#
# Fresh-process examples:
#   QC_FASTLOWER_MODE=fast QC_FASTLOWER_ORDER=3 \
#     julia --project=benchmark benchmark/fast_lowering_experiment.jl
#   QC_FASTLOWER_MODE=fast QC_FASTLOWER_ORDER=4 QC_FASTLOWER_VALIDATE=0 \
#     julia --project=benchmark benchmark/fast_lowering_experiment.jl
#   QC_FASTLOWER_MODE=baseline QC_FASTLOWER_ORDER=4 \
#     julia --project=benchmark benchmark/fast_lowering_experiment.jl

using OrdinaryDiffEqTsit5: Tsit5, solve
using QuantumCumulants
using SciMLBase: ODEProblem
using SymbolicUtils: SymbolicUtils
using Symbolics: Symbolics, @variables

const QC = QuantumCumulants
const N = 6
const SHORT_TSPAN = (0.0, 0.01)
const MODE = get(ENV, "QC_FASTLOWER_MODE", "fast")
const ORDER = parse(Int, get(ENV, "QC_FASTLOWER_ORDER", "3"))
const VALIDATE = get(ENV, "QC_FASTLOWER_VALIDATE", ORDER <= 3 ? "1" : "0") == "1"
const RESULTS = get(
    ENV,
    "QC_FASTLOWER_RESULTS",
    joinpath(@__DIR__, "results", "fast-lowering-$(MODE)-order$(ORDER).log"),
)

mkpath(dirname(RESULTS))
open(RESULTS, "w") do io
    println(io, "# package-load and precompilation time excluded")
end

function emit(line)
    println(line)
    open(RESULTS, "a") do io
        println(io, line)
    end
    flush(stdout)
    return line
end

function timed(f)
    GC.gc()
    result = @timed f()
    return result.value, result.time, result.bytes
end

function ising_model(order)
    h = ⊗([PauliSpace(Symbol(:spin, i)) for i in 1:N]...)
    σx(i) = Pauli(h, :σ, 1, i)
    σy(i) = Pauli(h, :σ, 2, i)
    σz(i) = Pauli(h, :σ, 3, i)
    σm(i) = (σx(i) - 1im * σy(i)) / 2
    @variables J hx γ
    H = -J * sum(σz(i) * σz(i + 1) for i in 1:(N - 1)) -
        hx * sum(σx(i) for i in 1:N)
    eqs = meanfield(
        [σz(i) for i in 1:N],
        H,
        [σm(i) for i in 1:N];
        rates = [γ for _ in 1:N],
        order,
    )
    return eqs, Dict(J => 1.0, hx => 1.0, γ => 0.2)
end

function nonzero_state(seed, n)
    return ComplexF64[
        0.11 * sin(seed + 0.17i) + 0.07im * cos(0.31seed + 0.11i) for i in 1:n
    ]
end

# --- Direct term extraction -----------------------------------------------------------

_unwrap(x::Symbolics.Num) = SymbolicUtils.unwrap(x)
_unwrap(x) = x

function _state_factor(x, idx)
    x = _unwrap(x)
    haskey(idx, x) && return idx[x]
    if x isa SymbolicUtils.BasicSymbolic && SymbolicUtils.iscall(x)
        op = SymbolicUtils.operation(x)
        args = SymbolicUtils.arguments(x)
        if op === conj && length(args) == 1
            y = _unwrap(args[1])
            haskey(idx, y) && return Int32(-idx[y])
        end
    end
    return nothing
end

function _contains_state(x, idx)
    x = _unwrap(x)
    _state_factor(x, idx) === nothing || return true
    x isa SymbolicUtils.BasicSymbolic || return false
    SymbolicUtils.iscall(x) || return false
    for a in SymbolicUtils.arguments(x)
        _contains_state(a, idx) && return true
    end
    return false
end

function _integer_exponent(e)
    e = _unwrap(e)
    value = if e isa Number
        e
    else
        try
            SymbolicUtils.unwrap_const(e)
        catch
            return nothing
        end
    end
    value isa Integer && return Int(value)
    value isa Rational && denominator(value) == 1 && return Int(value)
    value isa Real && isinteger(value) && return Int(value)
    return nothing
end

function _coefficient_product(parts::Vector{Any})
    isempty(parts) && return 1
    length(parts) == 1 && return parts[1]
    value = parts[1]
    @inbounds for i in 2:length(parts)
        value *= parts[i]
    end
    return value
end

# Parse one multiplicative state-polynomial term. State-free subtrees remain untouched and
# become coefficient factors. A state-dependent subtree is accepted only when it is a
# moment leaf, an integer power of one moment leaf, a product of accepted factors, or a
# quotient by a state-free denominator. Anything else asks the caller to use the generic
# polynomial_coeffs fallback for the whole equation.
function _parse_product!(factors::Vector{Int32}, coeffparts::Vector{Any}, x, idx)
    x = _unwrap(x)

    j = _state_factor(x, idx)
    if j !== nothing
        push!(factors, j)
        return true
    end

    if !_contains_state(x, idx)
        push!(coeffparts, x)
        return true
    end

    x isa SymbolicUtils.BasicSymbolic || return false
    SymbolicUtils.iscall(x) || return false
    op = SymbolicUtils.operation(x)
    args = SymbolicUtils.arguments(x)

    if op === (*)
        for a in args
            _parse_product!(factors, coeffparts, a, idx) || return false
        end
        return true
    elseif op === (^) && length(args) == 2
        j = _state_factor(args[1], idx)
        n = _integer_exponent(args[2])
        (j === nothing || n === nothing || n < 0) && return false
        for _ in 1:n
            push!(factors, j)
        end
        return true
    elseif op === (/) && length(args) == 2 && !_contains_state(args[2], idx)
        _parse_product!(factors, coeffparts, args[1], idx) || return false
        push!(coeffparts, inv(_unwrap(args[2])))
        return true
    elseif op === (-) && length(args) == 1
        push!(coeffparts, -1)
        return _parse_product!(factors, coeffparts, args[1], idx)
    end

    return false
end

function _append_sum_terms!(out::Vector{Any}, x)
    x = _unwrap(x)
    if x isa SymbolicUtils.BasicSymbolic && SymbolicUtils.iscall(x) &&
            SymbolicUtils.operation(x) === (+)
        for a in SymbolicUtils.arguments(x)
            _append_sum_terms!(out, a)
        end
    else
        push!(out, x)
    end
    return out
end

# Returns Vector{Tuple{Vector{Int32},Any}} or nothing when direct extraction cannot prove
# the expression belongs to the narrow coefficient×moment-monomial grammar.
function _direct_equation_terms(drift, idx)
    addterms = Any[]
    _append_sum_terms!(addterms, drift)
    result = Tuple{Vector{Int32}, Any}[]
    sizehint!(result, length(addterms))

    for term in addterms
        factors = Int32[]
        coeffparts = Any[]
        _parse_product!(factors, coeffparts, term, idx) || return nothing
        sort!(factors)
        push!(result, (factors, _coefficient_product(coeffparts)))
    end

    sort!(result; by = term -> Tuple(term[1]))
    return result
end

function _generic_equation_terms(drift, vars, idx, eqindex)
    dict, residual = Symbolics.polynomial_coeffs(drift, vars)
    QC._iszero_part(residual) || throw(QC.NonPolynomialDriftError(eqindex, residual))
    result = Tuple{Vector{Int32}, Any}[]
    sizehint!(result, length(dict))
    for (mono, coeff) in dict
        push!(result, (QC.monomial_factors(mono, idx), coeff))
    end
    sort!(result; by = term -> Tuple(term[1]))
    return result
end

function extract_terms(g, vars, idx)
    drifts = Any[Symbolics.unwrap(nd.drift) for nd in values(g.nodes)]
    extracted = Vector{Vector{Tuple{Vector{Int32}, Any}}}(undef, length(drifts))
    fallback = Int[]
    for i in eachindex(drifts)
        terms = _direct_equation_terms(drifts[i], idx)
        if terms === nothing
            push!(fallback, i)
            terms = _generic_equation_terms(drifts[i], vars, idx, i)
        end
        extracted[i] = terms
    end
    return extracted, fallback
end

# --- Allocation-light prefix interning -----------------------------------------------

function build_tables(termsets)
    # Monomial 1 is the empty product. Every other node is uniquely identified by the
    # already-interned parent plus one signed state factor.
    edges = Dict{Tuple{Int32, Int32}, Int32}()
    parent = Int32[0]
    leaf = Int32[0]
    coeff_ids = Dict{Any, Int32}()
    coeffs = Any[]
    coo_i = Int32[]
    coo_j = Int32[]
    coo_c = Int32[]

    function mono_id!(factors::Vector{Int32})
        p = Int32(1)
        @inbounds for factor in factors
            key = (p, factor)
            p = get!(edges, key) do
                push!(parent, p)
                push!(leaf, factor)
                Int32(length(parent))
            end
        end
        return p
    end

    nterms = sum(length, termsets)
    sizehint!(coo_i, nterms)
    sizehint!(coo_j, nterms)
    sizehint!(coo_c, nterms)

    for i in eachindex(termsets)
        for (factors, coeff) in termsets[i]
            j = mono_id!(factors)
            cid = get!(coeff_ids, coeff) do
                push!(coeffs, coeff)
                Int32(length(coeffs))
            end
            push!(coo_i, Int32(i))
            push!(coo_j, j)
            push!(coo_c, cid)
        end
    end

    return (; parent, leaf, coeffs, coo_i, coo_j, coo_c)
end

function fast_lower(eqs)
    (resolution, state_time, state_bytes) = timed(() -> QC.statevars_resolved(eqs))
    vars, idx = resolution

    (extraction, extraction_time, extraction_bytes) =
        timed(() -> extract_terms(eqs.graph, vars, idx))
    termsets, fallback = extraction

    (tables, table_time, table_bytes) = timed(() -> build_tables(termsets))
    iv = Symbolics.unwrap(eqs.iv)
    (params, params_time, params_bytes) = timed(() -> QC.discover_params(tables.coeffs, iv))

    ir = QC.MomentIR(
        length(eqs.graph.nodes),
        tables.parent,
        tables.leaf,
        tables.coeffs,
        tables.coo_i,
        tables.coo_j,
        tables.coo_c,
        params,
    )

    total_time = state_time + extraction_time + table_time + params_time
    total_bytes = state_bytes + extraction_bytes + table_bytes + params_bytes

    emit("stage state_resolution seconds=$state_time bytes=$state_bytes vars=$(length(vars))")
    emit("stage direct_extraction seconds=$extraction_time bytes=$extraction_bytes fallback_equations=$(length(fallback))")
    isempty(fallback) || emit("fallback_indices $(join(fallback, ','))")
    emit("stage table_build seconds=$table_time bytes=$table_bytes monomials=$(length(ir.parent)) coo=$(length(ir.coo_i)) coeffs=$(length(ir.coeffs))")
    emit("stage discover_params seconds=$params_time bytes=$params_bytes params=$(length(ir.params))")
    emit("fast_lower_total seconds=$total_time bytes=$total_bytes")

    return ir, fallback
end

function rhs_for_ir(eqs, ir, ps)
    kernel = QC.MomentKernel(ir; parallel = false)
    values = QC.kernel_pdict(ir.params, parameter_map(eqs, ps))
    parameters = QC.KernelParameters(ir, kernel.pattern, values)
    return QC._ode_function(QC.KernelRHS(kernel)), parameters
end

function validate_fast(eqs, fast_ir, ps)
    baseline_ir, baseline_time, baseline_bytes = timed(() -> QC._lower_moment_ir(eqs))
    emit("validation baseline_lower seconds=$baseline_time bytes=$baseline_bytes monomials=$(length(baseline_ir.parent)) coo=$(length(baseline_ir.coo_i))")

    fast_f, fast_p = rhs_for_ir(eqs, fast_ir, ps)
    baseline_f, baseline_p = rhs_for_ir(eqs, baseline_ir, ps)
    du_fast = zeros(ComplexF64, length(eqs.states))
    du_base = similar(du_fast)
    errors = Float64[]

    for seed in 1:3
        u = nonzero_state(seed, length(eqs.states))
        fast_f(du_fast, u, fast_p, 0.0)
        baseline_f(du_base, u, baseline_p, 0.0)
        push!(errors, maximum(abs.(du_fast .- du_base)))
    end

    key = first(keys(ps))
    update = Dict(key => ps[key] + 0.37)
    fast_updated = copy(fast_p)
    base_updated = copy(baseline_p)
    QC.update_parameters!(fast_updated, update)
    QC.update_parameters!(base_updated, update)
    u = nonzero_state(7, length(eqs.states))
    fast_f(du_fast, u, fast_updated, 0.0)
    baseline_f(du_base, u, base_updated, 0.0)
    updated_error = maximum(abs.(du_fast .- du_base))

    u0 = nonzero_state(11, length(eqs.states))
    fast_prob = ODEProblem(fast_f, u0, SHORT_TSPAN, fast_p)
    base_prob = ODEProblem(baseline_f, u0, SHORT_TSPAN, baseline_p)
    fast_sol = solve(fast_prob, Tsit5(); saveat = range(0.0, SHORT_TSPAN[2]; length = 11))
    base_sol = solve(base_prob, Tsit5(); saveat = range(0.0, SHORT_TSPAN[2]; length = 11))
    trajectory_error = maximum(
        maximum(abs.(a .- b)) for (a, b) in zip(fast_sol.u, base_sol.u)
    )

    emit("validation rhs_errors=$(join(errors, ','))")
    emit("validation updated_parameter_error=$updated_error")
    emit("validation trajectory_error=$trajectory_error")
    return maximum(errors; init = 0.0), updated_error, trajectory_error
end

function main()
    MODE in ("fast", "baseline") ||
        throw(ArgumentError("QC_FASTLOWER_MODE must be fast or baseline"))
    emit("metadata julia=$(VERSION) threads=$(Threads.nthreads()) order=$ORDER mode=$MODE validate=$VALIDATE")

    model, meanfield_time, meanfield_bytes = timed(() -> ising_model(ORDER))
    open_eqs, ps = model
    eqs, complete_time, complete_bytes = timed(() -> complete(open_eqs))
    emit("stage meanfield seconds=$meanfield_time bytes=$meanfield_bytes")
    emit("stage complete seconds=$complete_time bytes=$complete_bytes equations=$(length(eqs.states))")

    if MODE == "baseline"
        ir, lower_time, lower_bytes = timed(() -> QC._lower_moment_ir(eqs))
        emit("baseline_lower seconds=$lower_time bytes=$lower_bytes monomials=$(length(ir.parent)) coo=$(length(ir.coo_i)) coeffs=$(length(ir.coeffs))")
        return
    end

    fast_ir, fallback = fast_lower(eqs)
    if VALIDATE
        rhs_error, updated_error, trajectory_error = validate_fast(eqs, fast_ir, ps)
        (rhs_error == 0.0 && updated_error == 0.0 && trajectory_error == 0.0) ||
            error("fast lowering validation failed")
    end
    emit("result fallback_equations=$(length(fallback)) validation=$(VALIDATE ? "passed" : "skipped")")
    return
end

main()

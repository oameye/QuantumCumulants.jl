# Benchmark-only prototype for lowering completed QuantumCumulants drifts recursively to
# a sparse polynomial representation, without expanding every equation through
# Symbolics.polynomial_coeffs.
#
# Production source is intentionally untouched. Unsupported state-dependent operations
# fall back equation-by-equation to the existing Symbolics.polynomial_coeffs path.

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
    joinpath(@__DIR__, "results", "recursive-lowering-$(MODE)-order$(ORDER).log"),
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

# --- Recursive sparse-polynomial compiler -------------------------------------------

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

function _has_state(x, idx, cache::IdDict{Any, Bool})
    x = _unwrap(x)
    haskey(cache, x) && return cache[x]
    result = if _state_factor(x, idx) !== nothing
        true
    elseif !(x isa SymbolicUtils.BasicSymbolic) || !SymbolicUtils.iscall(x)
        false
    else
        any(a -> _has_state(a, idx, cache), SymbolicUtils.arguments(x))
    end
    cache[x] = result
    return result
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

_coeff_iszero(c) = try
    QC._iszero_part(c)
catch
    false
end

function _merge_monomials(a::Tuple, b::Tuple)
    isempty(a) && return b
    isempty(b) && return a
    out = Vector{Int32}(undef, length(a) + length(b))
    ia = 1
    ib = 1
    io = 1
    while ia <= length(a) && ib <= length(b)
        if a[ia] <= b[ib]
            out[io] = a[ia]
            ia += 1
        else
            out[io] = b[ib]
            ib += 1
        end
        io += 1
    end
    while ia <= length(a)
        out[io] = a[ia]
        ia += 1
        io += 1
    end
    while ib <= length(b)
        out[io] = b[ib]
        ib += 1
        io += 1
    end
    return Tuple(out)
end

function _poly_add_term!(out::Dict{Tuple, Any}, mono::Tuple, coeff)
    _coeff_iszero(coeff) && return out
    if haskey(out, mono)
        value = out[mono] + coeff
        if _coeff_iszero(value)
            delete!(out, mono)
        else
            out[mono] = value
        end
    else
        out[mono] = coeff
    end
    return out
end

function _poly_add!(out::Dict{Tuple, Any}, p::Dict{Tuple, Any})
    for (mono, coeff) in p
        _poly_add_term!(out, mono, coeff)
    end
    return out
end

function _poly_scale(p::Dict{Tuple, Any}, coeff)
    _coeff_iszero(coeff) && return Dict{Tuple, Any}()
    coeff == 1 && return p
    out = Dict{Tuple, Any}()
    sizehint!(out, length(p))
    for (mono, value) in p
        _poly_add_term!(out, mono, coeff * value)
    end
    return out
end

function _constant_coeff(p::Dict{Tuple, Any})
    length(p) == 1 || return nothing
    return get(p, (), nothing)
end

function _poly_mul(a::Dict{Tuple, Any}, b::Dict{Tuple, Any})
    isempty(a) && return Dict{Tuple, Any}()
    isempty(b) && return Dict{Tuple, Any}()
    ca = _constant_coeff(a)
    ca === nothing || return _poly_scale(b, ca)
    cb = _constant_coeff(b)
    cb === nothing || return _poly_scale(a, cb)

    out = Dict{Tuple, Any}()
    sizehint!(out, length(a) * length(b))
    for (ma, caa) in a, (mb, cbb) in b
        _poly_add_term!(out, _merge_monomials(ma, mb), caa * cbb)
    end
    return out
end

function _poly_pow(base::Dict{Tuple, Any}, n::Int)
    n < 0 && return nothing
    n == 0 && return Dict{Tuple, Any}(() => 1)
    n == 1 && return base
    result = Dict{Tuple, Any}(() => 1)
    power = base
    k = n
    while k > 0
        isodd(k) && (result = _poly_mul(result, power))
        k >>= 1
        k == 0 || (power = _poly_mul(power, power))
    end
    return result
end

function _reason_name(op)
    op === (+) && return :add
    op === (*) && return :mul
    op === (^) && return :pow
    op === (/) && return :div
    op === (-) && return :minus
    op === conj && return :conj
    return Symbol(string(op))
end

# Compile an arbitrary nested expression built from polynomial +, *, nonnegative integer
# powers and division by a state-free denominator. Any subtree with no moment-state leaves
# is retained intact as a coefficient, so indexed/scalar parameter structure is not rebuilt.
function _compile_poly(x, idx, state_cache::IdDict{Any, Bool}, reason::Base.RefValue{Any})
    x = _unwrap(x)

    j = _state_factor(x, idx)
    j === nothing || return Dict{Tuple, Any}((Int32(j),) => 1)

    if !_has_state(x, idx, state_cache)
        return Dict{Tuple, Any}(() => x)
    end

    if !(x isa SymbolicUtils.BasicSymbolic) || !SymbolicUtils.iscall(x)
        reason[] === nothing && (reason[] = :stateful_atom)
        return nothing
    end

    op = SymbolicUtils.operation(x)
    args = SymbolicUtils.arguments(x)

    if op === (+)
        out = Dict{Tuple, Any}()
        for arg in args
            p = _compile_poly(arg, idx, state_cache, reason)
            p === nothing && return nothing
            _poly_add!(out, p)
        end
        return out
    elseif op === (*)
        out = Dict{Tuple, Any}(() => 1)
        for arg in args
            p = _compile_poly(arg, idx, state_cache, reason)
            p === nothing && return nothing
            out = _poly_mul(out, p)
        end
        return out
    elseif op === (^) && length(args) == 2
        _has_state(args[2], idx, state_cache) && begin
            reason[] === nothing && (reason[] = :stateful_exponent)
            return nothing
        end
        n = _integer_exponent(args[2])
        if n === nothing || n < 0
            reason[] === nothing && (reason[] = :noninteger_power)
            return nothing
        end
        base = _compile_poly(args[1], idx, state_cache, reason)
        base === nothing && return nothing
        return _poly_pow(base, n)
    elseif op === (/) && length(args) == 2
        if _has_state(args[2], idx, state_cache)
            reason[] === nothing && (reason[] = :stateful_denominator)
            return nothing
        end
        numerator = _compile_poly(args[1], idx, state_cache, reason)
        numerator === nothing && return nothing
        return _poly_scale(numerator, inv(_unwrap(args[2])))
    elseif op === (-) && length(args) == 1
        p = _compile_poly(args[1], idx, state_cache, reason)
        p === nothing && return nothing
        return _poly_scale(p, -1)
    end

    reason[] === nothing && (reason[] = _reason_name(op))
    return nothing
end

function _generic_equation_terms(drift, vars, idx, eqindex)
    dict, residual = Symbolics.polynomial_coeffs(drift, vars)
    QC._iszero_part(residual) || throw(QC.NonPolynomialDriftError(eqindex, residual))
    result = Tuple{Tuple, Any}[]
    sizehint!(result, length(dict))
    for (mono, coeff) in dict
        push!(result, (Tuple(QC.monomial_factors(mono, idx)), coeff))
    end
    sort!(result; by = first)
    return result
end

function extract_terms(g, vars, idx)
    drifts = Any[Symbolics.unwrap(nd.drift) for nd in values(g.nodes)]
    extracted = Vector{Vector{Tuple{Tuple, Any}}}(undef, length(drifts))
    fallback = Int[]
    reasons = Dict{Any, Int}()
    state_cache = IdDict{Any, Bool}()

    for i in eachindex(drifts)
        reason = Ref{Any}(nothing)
        poly = _compile_poly(drifts[i], idx, state_cache, reason)
        if poly === nothing
            push!(fallback, i)
            r = reason[]
            reasons[r] = get(reasons, r, 0) + 1
            extracted[i] = _generic_equation_terms(drifts[i], vars, idx, i)
        else
            terms = Tuple{Tuple, Any}[(mono, coeff) for (mono, coeff) in poly]
            sort!(terms; by = first)
            extracted[i] = terms
        end
    end
    return extracted, fallback, reasons
end

# --- MomentIR table construction -----------------------------------------------------

function build_tables(termsets)
    edges = Dict{Tuple{Int32, Int32}, Int32}()
    parent = Int32[0]
    leaf = Int32[0]
    coeff_ids = Dict{Any, Int32}()
    coeffs = Any[]
    coo_i = Int32[]
    coo_j = Int32[]
    coo_c = Int32[]

    function mono_id!(factors::Tuple)
        p = Int32(1)
        @inbounds for factor in factors
            f = Int32(factor)
            key = (p, f)
            p = get!(edges, key) do
                push!(parent, p)
                push!(leaf, f)
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
            _coeff_iszero(coeff) && continue
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
    resolution, state_time, state_bytes = timed(() -> QC.statevars_resolved(eqs))
    vars, idx = resolution

    extraction, extraction_time, extraction_bytes =
        timed(() -> extract_terms(eqs.graph, vars, idx))
    termsets, fallback, reasons = extraction

    tables, table_time, table_bytes = timed(() -> build_tables(termsets))
    iv = Symbolics.unwrap(eqs.iv)
    params, params_time, params_bytes = timed(() -> QC.discover_params(tables.coeffs, iv))

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
    emit("stage recursive_extraction seconds=$extraction_time bytes=$extraction_bytes fallback_equations=$(length(fallback))")
    isempty(fallback) || emit("fallback_indices $(join(fallback, ','))")
    isempty(reasons) || emit("fallback_reasons $(join([string(k, ':', v) for (k, v) in sort!(collect(reasons); by = first)], ','))")
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

function monomial_support(ir)
    support = Vector{Tuple}(undef, length(ir.parent))
    support[1] = ()
    for m in 2:length(ir.parent)
        support[m] = (support[ir.parent[m]]..., ir.leaf[m])
    end
    return Set(support)
end

function validate_fast(eqs, fast_ir, ps)
    baseline_ir, baseline_time, baseline_bytes = timed(() -> QC._lower_moment_ir(eqs))
    emit("validation baseline_lower seconds=$baseline_time bytes=$baseline_bytes monomials=$(length(baseline_ir.parent)) coo=$(length(baseline_ir.coo_i))")
    emit("validation support_equal=$(monomial_support(fast_ir) == monomial_support(baseline_ir))")

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
        return nothing
    end

    fast_ir, fallback = fast_lower(eqs)
    if VALIDATE
        rhs_error, parameter_error, trajectory_error = validate_fast(eqs, fast_ir, ps)
        maximum((rhs_error, parameter_error, trajectory_error)) <= 1.0e-11 ||
            error("recursive lowering validation failed")
    end
    emit("result fallback_equations=$(length(fallback))")
    return nothing
end

main()

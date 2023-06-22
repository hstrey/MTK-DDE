using DiffEqBase, SciMLBase
using ModelingToolkit
using ModelingToolkit: process_DEProblem, process_events, has_discrete_subsystems, build_explicit_observed_function, get_discrete_subsystems, get_metadata, get_iv, 
using SciMLBase: StandardODEProblem
using Symbolics: unwrap
using SymbolicUtils
using SymbolicUtils.Code
using ModelingToolkit: isvariable
using LinearAlgebra: I

function generate_ddefunction(sys::ModelingToolkit.AbstractODESystem, dvs = states(sys),
    ps = parameters(sys); kwargs...)
    iv = unwrap(only(independent_variables(sys)))
    eqs = equations(sys)

    varss = Set()
    for eq in eqs
        ModelingToolkit.vars!(varss, eq)
    end
    varss

    delay_terms = filter(varss) do v
        isvariable(v) || return false
        istree(v) || return false
        if operation(v) === getindex
            v = arguments(v)[1]
        end
        istree(v) || return false
        args = arguments(v)
        length(args) == 1 && !isequal(iv, args[1]) && occursin(iv, args[1])
    end |> collect

    hh = Sym{Any}(:h)
    op_states = operation.(dvs)

    eqs2 = copy(eqs)
    for d in delay_terms
        # find idx of state that has delay
        stidx = findfirst(isequal(operation(d)), op_states)
        if !isnothing(stidx)
                eqs2 = substitute.(eqs2,
                    (Dict(d => term(getindex,
                            term(hh, ps, unwrap(arguments(d)[1]), type = Real),
                                                stidx[1], type = Real)),))
        end
    end

    out = Sym{Any}(:out)
    body = SetArray(false, out, getfield.(eqs2, :rhs))
    func = Func([out, DestructuredArgs(dvs), hh, DestructuredArgs(ps), iv],
                [], body)
    my_func_expr = toexpr(func)
    eval(my_func_expr)
end

"""
```julia
DiffEqBase.DDEFunction{iip}(sys::AbstractODESystem, dvs = states(sys),
                            ps = parameters(sys);
                            version = nothing, tgrad = false,
                            jac = false,
                            sparse = false,
                            kwargs...) where {iip}
```

Create an `DDEFunction` from the [`System`](@ref). The arguments `dvs` and `ps`
are used to set the order of the dependent variable and parameter vectors,
respectively.
"""
function DiffEqBase.DDEFunction(sys::ModelingToolkit.AbstractODESystem, args...; kwargs...)
    DDEFunction{true}(sys, args...; kwargs...)
end

function DiffEqBase.DDEFunction{true}(sys::ModelingToolkit.AbstractODESystem, args...;
    kwargs...)
    DDEFunction{true, SciMLBase.AutoSpecialize}(sys, args...; kwargs...)
end

function DiffEqBase.DDEFunction{false}(sys::ModelingToolkit.AbstractODESystem, args...;
    kwargs...)
    DDEFunction{false, SciMLBase.FullSpecialize}(sys, args...; kwargs...)
end

function MyDDEFunction(sys::ModelingToolkit.AbstractODESystem, dvs = states(sys),
    ps = parameters(sys), u0 = nothing;
    version = nothing, tgrad = false,
    jac = false, p = nothing,
    t = nothing,
    eval_expression = true,
    sparse = false, simplify = false,
    eval_module = @__MODULE__,
    steady_state = false,
    checkbounds = false,
    sparsity = false,
    analytic = nothing,
    kwargs...)

    f = generate_ddefunction(sys, states(sys), parameters(sys);
        kwargs...)

    if tgrad
        tgrad_gen = generate_tgrad(sys, dvs, ps;
            simplify = simplify,
            expression = Val{eval_expression},
            expression_module = eval_module,
            checkbounds = checkbounds, kwargs...)
        tgrad_oop, tgrad_iip = tgrad_gen
        _tgrad(u, p, t) = tgrad_oop(u, p, t)
        _tgrad(J, u, p, t) = tgrad_iip(J, u, p, t)
    else
        _tgrad = nothing
    end

    if jac
        jac_gen = generate_jacobian(sys, dvs, ps;
            simplify = simplify, sparse = sparse,
            expression = Val{eval_expression},
            expression_module = eval_module,
            checkbounds = checkbounds, kwargs...)
        jac_oop, jac_iip = jac_gen
        _jac(u, p, t) = jac_oop(u, p, t)
        _jac(J, u, p, t) = jac_iip(J, u, p, t)
    else
        _jac = nothing
    end

    M = calculate_massmatrix(sys)

    _M = if sparse && !(u0 === nothing || M === I)
        SparseArrays.sparse(M)
    elseif u0 === nothing || M === I
        M
    else
        ArrayInterface.restructure(u0 .* u0', M)
    end

    obs = observed(sys)
    observedfun = if steady_state
        let sys = sys, dict = Dict()
            function generated_observed(obsvar, args...)
                obs = get!(dict, value(obsvar)) do
                    build_explicit_observed_function(sys, obsvar)
                end
                if args === ()
                    let obs = obs
                        (u, p, t = Inf) -> obs(u, p, t)
                    end
                else
                    length(args) == 2 ? obs(args..., Inf) : obs(args...)
                end
            end
        end
    else
        let sys = sys, dict = Dict()
            function generated_observed(obsvar, args...)
                obs = get!(dict, value(obsvar)) do
                    build_explicit_observed_function(sys, obsvar; checkbounds = checkbounds)
                end
                if args === ()
                    let obs = obs
                        (u, p, t) -> obs(u, p, t)
                    end
                else
                    obs(args...)
                end
            end
        end
    end

    jac_prototype = if sparse
        uElType = u0 === nothing ? Float64 : eltype(u0)
        if jac
            similar(calculate_jacobian(sys, sparse = sparse), uElType)
        else
            similar(jacobian_sparsity(sys), uElType)
        end
    else
        nothing
    end

    DDEFunction(f;
        sys = sys,
        jac = _jac === nothing ? nothing : _jac,
        tgrad = _tgrad === nothing ? nothing : _tgrad,
        mass_matrix = _M,
        jac_prototype = jac_prototype,
        syms = Symbol.(states(sys)),
        indepsym = Symbol(get_iv(sys)),
        paramsyms = Symbol.(ps),
        observed = observedfun,
        sparsity = sparsity ? jacobian_sparsity(sys) : nothing,
        analytic = analytic)
end

function MyDDEProblem(sys::ModelingToolkit.AbstractODESystem, u0map = [],
    h = nothing,
    tspan = get_tspan(sys),
    parammap = DiffEqBase.NullParameters();
    callback = nothing,
    check_length = true,
    kwargs...)
    has_difference = false
    f, u0, p = process_DEProblem(MyDDEFunction, sys, u0map, parammap;
        t = tspan !== nothing ? tspan[1] : tspan,
        check_length, kwargs...)
    @show u0
    @show p
    cbs = process_events(sys; callback, has_difference, kwargs...)
    if has_discrete_subsystems(sys) && (dss = get_discrete_subsystems(sys)) !== nothing
        affects, clocks, svs = ModelingToolkit.generate_discrete_affect(dss...)
        discrete_cbs = map(affects, clocks, svs) do affect, clock, sv
            if clock isa Clock
                PeriodicCallback(DiscreteSaveAffect(affect, sv), clock.dt)
            else
                error("$clock is not a supported clock type.")
            end
        end
        if cbs === nothing
            if length(discrete_cbs) == 1
                cbs = only(discrete_cbs)
            else
                cbs = CallbackSet(discrete_cbs...)
            end
        else
            cbs = CallbackSet(cbs, discrete_cbs)
        end
    else
        svs = nothing
    end
    kwargs = filter_kwargs(kwargs)
    pt = something(get_metadata(sys), StandardODEProblem())

    kwargs1 = (;)
    if cbs !== nothing
        kwargs1 = merge(kwargs1, (callback = cbs,))
    end
    if svs !== nothing
        kwargs1 = merge(kwargs1, (disc_saved_values = svs,))
    end
    DDEProblem(f, u0, h, tspan, p, pt; kwargs1..., kwargs...)
end
get_callback(prob::DDEProblem) = prob.kwargs[:callback]

function generate_function(sys::AbstractODESystem, dvs = states(sys), ps = parameters(sys);
                           implicit_dae = false,
                           ddvs = implicit_dae ? map(Differential(get_iv(sys)), dvs) :
                                  nothing,
                           has_difference = false,
                           kwargs...)
    eqs = [eq for eq in equations(sys) if !isdifferenceeq(eq)]
    if !implicit_dae
        check_operator_variables(eqs, Differential)
        check_lhs(eqs, Differential, Set(dvs))
    end
    # substitute x(t) by just x
    rhss = implicit_dae ? [_iszero(eq.lhs) ? eq.rhs : eq.rhs - eq.lhs for eq in eqs] :
           [eq.rhs for eq in eqs]

    # TODO: add an optional check on the ordering of observed equations
    u = map(x -> time_varying_as_func(value(x), sys), dvs)
    p = map(x -> time_varying_as_func(value(x), sys), ps)
    t = get_iv(sys)

    pre, sol_states = get_substitutions_and_solved_states(sys,
                                                          no_postprocess = has_difference)

    if implicit_dae
        build_function(rhss, ddvs, u, p, t; postprocess_fbody = pre, states = sol_states,
                       kwargs...)
    else
        build_function(rhss, u, p, t; postprocess_fbody = pre, states = sol_states,
                       kwargs...)
    end
end


"""
    u0, p, defs = get_u0_p(sys, u0map, parammap; use_union=false, tofloat=!use_union)

Take dictionaries with initial conditions and parameters and convert them to numeric arrays `u0` and `p`. Also return the merged dictionary `defs` containing the entire operating point. 
"""
function get_u0_p(sys, u0map, parammap; use_union = false, tofloat = !use_union)
    eqs = equations(sys)
    dvs = states(sys)
    ps = parameters(sys)

    defs = defaults(sys)
    defs = mergedefaults(defs, parammap, ps)
    defs = mergedefaults(defs, u0map, dvs)

    u0 = varmap_to_vars(u0map, dvs; defaults = defs, tofloat = true)
    p = varmap_to_vars(parammap, ps; defaults = defs, tofloat, use_union)
    p = p === nothing ? SciMLBase.NullParameters() : p
    u0, p, defs
end

function process_DEProblem(constructor, sys::AbstractODESystem, u0map, parammap;
                           implicit_dae = false, du0map = nothing,
                           version = nothing, tgrad = false,
                           jac = false,
                           checkbounds = false, sparse = false,
                           simplify = false,
                           linenumbers = true, parallel = SerialForm(),
                           eval_expression = true,
                           use_union = false,
                           tofloat = !use_union,
                           kwargs...)
    eqs = equations(sys)
    dvs = states(sys)
    ps = parameters(sys)
    iv = get_iv(sys)

    u0, p, defs = get_u0_p(sys, u0map, parammap; tofloat, use_union)

    if implicit_dae && du0map !== nothing
        ddvs = map(Differential(iv), dvs)
        defs = mergedefaults(defs, du0map, ddvs)
        du0 = varmap_to_vars(du0map, ddvs; defaults = defs, toterm = identity,
                             tofloat = true)
    else
        du0 = nothing
        ddvs = nothing
    end

    check_eqs_u0(eqs, dvs, u0; kwargs...)

    f = constructor(sys, dvs, ps, u0; ddvs = ddvs, tgrad = tgrad, jac = jac,
                    checkbounds = checkbounds, p = p,
                    linenumbers = linenumbers, parallel = parallel, simplify = simplify,
                    sparse = sparse, eval_expression = eval_expression, kwargs...)
    implicit_dae ? (f, du0, u0, p) : (f, u0, p)
end


"""
```julia
DiffEqBase.ODEFunction{iip}(sys::AbstractODESystem, dvs = states(sys),
                            ps = parameters(sys);
                            version = nothing, tgrad = false,
                            jac = false,
                            sparse = false,
                            kwargs...) where {iip}
```

Create an `ODEFunction` from the [`ODESystem`](@ref). The arguments `dvs` and `ps`
are used to set the order of the dependent variable and parameter vectors,
respectively.
"""
function DiffEqBase.ODEFunction(sys::AbstractODESystem, args...; kwargs...)
    ODEFunction{true}(sys, args...; kwargs...)
end

function DiffEqBase.ODEFunction{true}(sys::AbstractODESystem, args...;
                                      kwargs...)
    ODEFunction{true, SciMLBase.AutoSpecialize}(sys, args...; kwargs...)
end

function DiffEqBase.ODEFunction{false}(sys::AbstractODESystem, args...;
                                       kwargs...)
    ODEFunction{false, SciMLBase.FullSpecialize}(sys, args...; kwargs...)
end

function DiffEqBase.ODEFunction{iip, specialize}(sys::AbstractODESystem, dvs = states(sys),
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
                                                 kwargs...) where {iip, specialize}
    f_gen = generate_function(sys, dvs, ps; expression = Val{eval_expression},
                              expression_module = eval_module, checkbounds = checkbounds,
                              kwargs...)
    f_oop, f_iip = eval_expression ?
                   (@RuntimeGeneratedFunction(eval_module, ex) for ex in f_gen) : f_gen
    f(u, p, t) = f_oop(u, p, t)
    f(du, u, p, t) = f_iip(du, u, p, t)

    if specialize === SciMLBase.FunctionWrapperSpecialize && iip
        if u0 === nothing || p === nothing || t === nothing
            error("u0, p, and t must be specified for FunctionWrapperSpecialize on ODEFunction.")
        end
        f = SciMLBase.wrapfun_iip(f, (u0, u0, p, t))
    end

    if tgrad
        tgrad_gen = generate_tgrad(sys, dvs, ps;
                                   simplify = simplify,
                                   expression = Val{eval_expression},
                                   expression_module = eval_module,
                                   checkbounds = checkbounds, kwargs...)
        tgrad_oop, tgrad_iip = eval_expression ?
                               (@RuntimeGeneratedFunction(eval_module, ex) for ex in tgrad_gen) :
                               tgrad_gen
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
        jac_oop, jac_iip = eval_expression ?
                           (@RuntimeGeneratedFunction(eval_module, ex) for ex in jac_gen) :
                           jac_gen
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

    ODEFunction{iip, specialize}(f;
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


"""
```julia
DiffEqBase.ODEProblem{iip}(sys::AbstractODESystem, u0map, tspan,
                           parammap = DiffEqBase.NullParameters();
                           version = nothing, tgrad = false,
                           jac = false,
                           checkbounds = false, sparse = false,
                           simplify = false,
                           linenumbers = true, parallel = SerialForm(),
                           kwargs...) where {iip}
```

Generates an ODEProblem from an ODESystem and allows for automatically
symbolically calculating numerical enhancements.
"""
function DiffEqBase.ODEProblem(sys::AbstractODESystem, args...; kwargs...)
    ODEProblem{true}(sys, args...; kwargs...)
end

function DiffEqBase.ODEProblem{true}(sys::AbstractODESystem, args...; kwargs...)
    ODEProblem{true, SciMLBase.AutoSpecialize}(sys, args...; kwargs...)
end

function DiffEqBase.ODEProblem{false}(sys::AbstractODESystem, args...; kwargs...)
    ODEProblem{false, SciMLBase.FullSpecialize}(sys, args...; kwargs...)
end

struct DiscreteSaveAffect{F, S} <: Function
    f::F
    s::S
end
(d::DiscreteSaveAffect)(args...) = d.f(args..., d.s)

function DiffEqBase.ODEProblem{iip, specialize}(sys::AbstractODESystem, u0map = [],
                                                tspan = get_tspan(sys),
                                                parammap = DiffEqBase.NullParameters();
                                                callback = nothing,
                                                check_length = true,
                                                kwargs...) where {iip, specialize}
    has_difference = any(isdifferenceeq, equations(sys))
    f, u0, p = process_DEProblem(ODEFunction{iip, specialize}, sys, u0map, parammap;
                                 t = tspan !== nothing ? tspan[1] : tspan,
                                 has_difference = has_difference,
                                 check_length, kwargs...)
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
    ODEProblem{iip}(f, u0, tspan, p, pt; kwargs1..., kwargs...)
end
get_callback(prob::ODEProblem) = prob.kwargs[:callback]

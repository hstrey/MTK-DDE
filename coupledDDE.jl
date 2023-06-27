using ModelingToolkit, DifferentialEquations, Plots
using ModelingToolkit: process_DEProblem, process_events, has_discrete_subsystems, build_explicit_observed_function, get_discrete_subsystems, get_metadata, get_iv, filter_kwargs, get_u0_p
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
    @show eqs2
    out = Sym{Any}(:out)
    body = SetArray(false, out, getfield.(eqs2, :rhs))
    func = Func([out, DestructuredArgs(dvs), hh, DestructuredArgs(ps), iv],
                [], body)
    my_func_expr = toexpr(func)
    eval(my_func_expr)
end

@variables t
D = Differential(t)

function van_der_pol(;name, θ=1.0, τ=0.01)
    @parameters θ=θ τ=τ
    @variables x(..) y(..) jcn(t)=0.0
    histx1 = x(t-τ)
    x = x(t)
    y = y(t)
    eqs = [D(x) ~ y + jcn,
           D(y) ~ θ*(1-histx1^2)*y - x]

    return System(eqs, t, [x,y,jcn], [θ,τ]; name=name)
end

@named VP1 = van_der_pol()
@named VP2 = van_der_pol()

st1 = states.((VP1,), states(VP1))
st2 = states.((VP2,), states(VP2))

eqs = [VP1.jcn ~ VP2.x,
        VP2.jcn ~ VP1.x]

sys = [VP1,VP2]

@named connected = System(eqs,t)
@named coupledVP = compose(System(eqs,t;name=:connected),sys)

# run structural simplify with consistency check off
coupledVPs = structural_simplify(coupledVP; check_consistency=false)
eqs = equations(coupledVPs)
sts = states(coupledVPs)[1:4]
iv = independent_variables(coupledVPs)
ps = parameters(coupledVPs)

f = generate_ddefunction(coupledVPs2)
tspan = (0.0, 20.0)
h(p, t) = ones(4)

prob = DDEProblem(f,[0.5,0.4,0.5,0.4],h,tspan,[1.0,0.01,1.0,0.01];constant_lags = [0.01,0.01])
alg = MethodOfSteps(Tsit5())
sol = solve(prob, alg)

plot(sol, title="Coupled van der Pol DDEs")
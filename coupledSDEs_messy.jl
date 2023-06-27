using ModelingToolkit, DifferentialEquations, Plots
using ModelingToolkit: TearingState, vars, is_delay_var, isvariable, VariableSource
using SymbolicUtils
using SymbolicUtils: istree, arguments, operation, similarterm, promote_symtype,
    Symbolic, isadd, ismul, ispow, issym, FnType,
    @rule, Rewriters, substitute, metadata, BasicSymbolic,
    Sym, Term
using Symbolics
using Symbolics: rename, get_variables!, _solve, hessian_sparsity,
    jacobian_sparsity, isaffine, islinear, _iszero, _isone,
    tosymbol, lower_varname, diff2term, var_from_nested_derivative,
    BuildTargets, JuliaTarget, StanTarget, CTarget, MATLABTarget,
    ParallelForm, SerialForm, MultithreadedForm, build_function,
    rhss, lhss, prettify_expr, gradient,
    jacobian, hessian, derivative, sparsejacobian, sparsehessian,
    substituter, scalarize, getparent
    
@variables t
D = Differential(t)

myisvariable(x::Num)::Bool = myisvariable(value(x))
function myisvariable(x)::Bool
    x isa Symbolic || return false
    p = getparent(x, nothing)
    @show p
    p === nothing || (x = p)
    @show p
    hasmetadata(x, VariableSource)
end

function van_der_pol(;name, θ=1.0, τ=0.01)
    @parameters θ=θ τ=τ ϕ=ϕ
    @variables x(..) y(..) jcn(t)=0.0
    histx1 = x(t-τ)
    x = x(t)
    y = y(t)
    eqs = [D(x) ~ y + jcn,
           D(y) ~ θ*(1-histx1^2)*y - x]

    return System(eqs, t, [x,y,jcn], [θ,τ,ϕ]; name=name)
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

eqs = equations(coupledVP)
v = vars(eqs[4])

for vv in v
    @show myisvariable(vv)
    if istree(vv)
        @show operation(vv)
        vvv = arguments(vv)
    end
end

TS = TearingState(coupledVP)

coupledVPs = structural_simplify(coupledVP; check_consistency=false)

prob = SDEProblem(coupledVPs, [], (0, 2.0))
sol = solve(prob)
plot(sol, title="Coupled van der Pol SDEs")
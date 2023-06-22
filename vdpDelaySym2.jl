using ModelingToolkit
using Symbolics: unwrap
using SymbolicUtils
using SymbolicUtils.Code
using ModelingToolkit: isvariable
using DifferentialEquations
using Plots

pars = @parameters θ, τ1, τ2
@variables t x(..) y(..)
D = Differential(t)
histx1 = x(t-τ1)
histx2 = x(t-0.1)
histy = y(t-τ2)
x = x(t)
y = y(t)
eqs = [D(x) ~ histy,
       D(y) ~ θ*(1-histx1^2)*y - histx2]

@named vdpDelayODE = System(eqs,t,[x,y],[θ, τ1,τ2])

iv = unwrap(only(independent_variables(vdpDelayODE)))
eqs = equations(vdpDelayODE)
p = parameters(vdpDelayODE)
s = states(vdpDelayODE)

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
op_states = operation.(s)

eqs2 = copy(eqs)
for d in delay_terms
       # find idx of state that has delay
       stidx = findfirst(isequal(operation(d)), op_states)
       if !isnothing(stidx)
              eqs2 = substitute.(eqs2,
                  (Dict(d => term(getindex,
                        term(hh, p, unwrap(arguments(d)[1]), type = Real),
                                               stidx[1], type = Real)),))
       end
end

out = Sym{Any}(:out)
body = SetArray(false, out, getfield.(eqs2, :rhs))
func = Func([out, DestructuredArgs(states(vdpDelayODE)), hh, DestructuredArgs(parameters(vdpDelayODE)), iv],
            [], body)
my_func_expr = toexpr(func)
my_func = eval(my_func_expr)

u0map = [
           x => 0.1,
           y => 0.1
        ]
       
parammap = [
           θ => 1.0,
           τ1 => 0.1,
           τ2 => 0.01
           ]

tspan = (0.0, 20.0)
h(p, t) = ones(2)
prob = DDEProblem(vdpDelayODE,u0map,h,(0,100),parammap;constant_lags = [0.1,0.01])

alg = MethodOfSteps(Rodas4())
sol = solve(prob, alg)

plot(sol)

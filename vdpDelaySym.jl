using ModelingToolkit

@parameters θ, ϕ
@variables t x(t) y(t)
D = Differential(t)

@delays τ

eqs = [D(x) ~ y,
       D(y) ~ θ*(1-x(t-τ)^2*y) - x(t-τ)]

@named vdpDelay = System(eqs,t)

u0map = [
           x => 0.1,
           y => 0.1
        ]
       
parammap = [
           θ => 1.0,
           ϕ => 0.1
           ]

lags = [ τ => 0.1 ]
       
tspan = [0.0, 20]

prob = DDEProblem(vdpDelay,u0map,tspan,parammap;constant_lags = lags)
solve(prob)


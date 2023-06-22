using ModelingToolkit, DifferentialEquations, Plots

@variables t
D = Differential(t)

function van_der_pol(;name, θ=1.0, τ=0.01, ϕ=1.0)
    @parameters θ=θ, τ=τ, ϕ=ϕ
    @variables x(..) y(..) jcn(t)=0.0
    @brownian ξ
    histx1 = x(t-τ1)
    x = x(t)
    y = y(t)
    eqs = [D(x) ~ y + jcn + ϕ*ξ,
           D(y) ~ θ*(1-hist1^2)*y - x + ϕ*ξ]

    return System(eqs, t; name=name)
end

@named VP1 = van_der_pol()
@named VP2 = van_der_pol()

eqs = [VP1.jcn ~ VP2.x,
        VP2.jcn ~ VP1.x]

sys = [VP1,VP2]

@named connected = System(eqs,t)
@named coupledVP = compose(System(eqs,t;name=:connected),sys)
coupledVPs = structural_simplify(coupledVP)

prob = SDEProblem(coupledVPs, [], (0, 2.0))
sol = solve(prob)
plot(sol, title="Coupled van der Pol SDEs")
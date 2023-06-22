using DifferentialEquations
using Plots

function two_nodes(du, u, p, t)
    μ1, μ2, k11, k22, k33, k44 = p
    #hist1 = h(p, t - τ1)[1]
    #hist2 = h(p, t - τ2)[2]
    #hist3 = h(p, t - τ3)[3]
    du[1] = k11*u[1] + u[3]*u[2]
    du[2] = k22*u[2] + u[4]*u[1]
    du[3] = k33*(u[3]-μ1)
    du[4] = k44*(u[4]-μ2)
end

function noise(du, u, p, t)
    du[1] = 1.0
    du[2] = 1.0
    du[3] = 0.1
    du[4] = 0.1
end

p = [-1.0,-2.0,-10,-10,-0.2,-0.2]
u0 = [0,0,-1,-2]

prob = SDEProblem(two_nodes,noise, u0,(0,100),p)
sol = solve(prob,SRIW1())

plot(sol)
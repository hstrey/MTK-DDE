# MTK-DDE
ModelingToolkit implementation of Delay Differential Equations

Implementation has the following steps:
- create delays as parameters
- write symbolic equations
- create ODESystem
- extract iv, eqs, states, variables
- extract initial conditions, parameters
- find delay terms
- create history function h
- create set of equations with delay terms being substituted by h function
- create executable function f(du,u,h,p)
- create DDEProblem

MTK steps to turn symbolic system into ODEProblem

MTK.abstractodesystem.jl
- call ODEProblem on ODESystem
- call process_DEProblem(constructor = ODEFunction, ...)
- call get_u0_p
- construct function by calling ODEfunction
Symbolics.build_function.jl
- call build_function


The challenge is to turn a symbolic list of states into u[1..n] then calculate the du[1..n] with parameters from a symbolic dictionary with u0 also from a dictionary.


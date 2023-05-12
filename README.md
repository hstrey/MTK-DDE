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




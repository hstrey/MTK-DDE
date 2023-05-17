function DDEFunction{iip, specialize}(sys::AbstractODESystem, dvs = states(sys),
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
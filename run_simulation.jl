task_id = parse(Int, ARGS[1])  # SLURM_ARRAY_TASK_ID

using Pkg
Pkg.activate(".")


using DataFrames
using Random
using JLD2
using MosekTools
using JuMP
using ProgressMeter
using StatsBase
using Distributions
using Empirikos
using MultipleTesting
using Gurobi 

dir = @__DIR__
include(joinpath(dir, "utils.jl"))
include(joinpath(dir, "methods.jl"))



# Monte Carlo replicates
nreps = 5


Random.seed!(1)


# settings

# Hard-coded for now 
n = 2_000 
n0 = 1_800
α = 0.1 

Ks = [5]

effect_sizes = [6.0]


variance_settings = (
    Dirac = Dirac(1.0),
    Uniform = Uniform(0.5, 2.0),
)

variance_keys = keys(variance_settings)




key_combinations = collect(Iterators.product(variance_keys, effect_sizes, Ks))

variance_key, effect_size, K = key_combinations[task_id]


ground_truth_variance_prior = getproperty(variance_settings, variance_key)


mosek_attrib = optimizer_with_attributes(
    Mosek.Optimizer,
    "QUIET" => true,
)


method_res = DataFrame(
    variance_setting = Symbol[],
    effect_size = Float64[],
    K = Int64[],
    simulation_number = Int64[],
    method_name = Symbol[],
    FDP = Float64[],
    Power = Float64[],
    discoveries = Int64[],
    α = Float64[],
)

𝒢 = DiscretePriorClass(exp.(range(start = log(0.001), stop = log(1000), length = 600)))
npmle_G = NPMLE(𝒢, mosek_attrib)




@showprogress dt = 1 desc = "iters" for k in Base.OneTo(nreps)
    @show k
    method_list = (
        ztest = ZTestEvalues(),
        oracle = OracleEvalues(),
        ttest  = TTestEvalues(),
        universal = UniversalInferenceEvalues(),
        plugin = PluginMixtureEvalues(),
        flocalized = FLocalizedMixtureEvalues(),
    )


    μ_alt = effect_size / sqrt(K)
    μs = [zeros(n0); fill(μ_alt, n-n0)]

    Hs = .! iszero.(μs)
    σs² = rand(ground_truth_variance_prior, n)
    σs = sqrt.(σs²)

    Zs = (rand(Normal(), n, K) .+ μs) .* σs
    zbar = vec(mean(Zs, dims=2))*sqrt(K)
    S² = vec(var(Zs, corrected=true, dims=2))

    Zs = NormalChiSquareSample.(zbar, S², K-1)
    S² = ScaledChiSquareSample.(Zs)

    Ĝ = fit(npmle_G,  S²).prior

    if isa(ground_truth_variance_prior, Dirac) 
        G_oracle = ground_truth_variance_prior
    else 
        G_oracle = DiscreteNonParametric(σs², fill(1/n, n) )
    end

    for key in keys(method_list)
        _method = getproperty(method_list, key)
        _apply_method = _method(Zs, α, effect_size, Ĝ=Ĝ, 𝒢=𝒢, G_oracle=G_oracle, σs=σs)
        _method_eval = evaluate_method(Hs, _apply_method)

        push!(
            method_res,
            (
                variance_setting = variance_key,
                effect_size = effect_size,
                K = K,
                simulation_number = k,
                method_name = key,
                _method_eval...,
                α = α,
            ),
        )
    end
end


store_name = joinpath(dir, "simulation_results", "method_res_$(variance_key)_$(effect_size)_K$(K).jld2")
jldsave(
    store_name;
    method_res = method_res,
)


#  Methods 

# Throughout simulations assume that 

using Gurobi
using Empirikos
using JuMP

struct OracleEvalues end 
struct ZTestEvalues end 
struct TTestEvalues end 
struct TTestPvalues end 
struct UniversalInferenceEvalues end 
struct PluginMixtureEvalues end 
struct FLocalizedMixtureEvalues end 




function (::ZTestEvalues)(Zs, α, effect_size; Ĝ=nothing, 𝒢=nothing,G_oracle=nothing,σs=nothing)
    Zs_normal = NormalSample.(getproperty.(Zs, :Z), σs)
    
    evals = pdf.(Dirac.(effect_size .* σs), Zs_normal) ./ pdf.(Dirac(0), Zs_normal)

    adjusted_evals = adjust(min.(1 ./ evals, 1), BenjaminiHochberg())
    rjs_idx = adjusted_evals .<= α
    (evalues=evals, rjs_idx=rjs_idx)
end 

function (::OracleEvalues)(Zs, α, effect_size; Ĝ=nothing, 𝒢=nothing,G_oracle=nothing,σs=nothing)
    denominator_dbn = product_distribution((λ=Dirac(0.0), σ²=G_oracle))
    numerator_dbn = product_distribution((λ=Dirac(effect_size), σ²=G_oracle))
    mixture_evalue = Empirikos.MixtureEValue(numerator_dbn, denominator_dbn)
    evals = mixture_evalue.(Zs)

    adjusted_evals = adjust(min.(1 ./ evals, 1), BenjaminiHochberg())
    rjs_idx = adjusted_evals .<= α
    (evalues=evals, rjs_idx=rjs_idx)
end 

function (::PluginMixtureEvalues)(Zs, α, effect_size; Ĝ=nothing, 𝒢=nothing,G_oracle=nothing,σs=nothing)
    denominator_dbn = product_distribution((λ=Dirac(0.0), σ²=Ĝ))
    numerator_dbn = product_distribution((λ=Dirac(effect_size), σ²=Ĝ))
    mixture_evalue = Empirikos.MixtureEValue(numerator_dbn, denominator_dbn)
    evals = mixture_evalue.(Zs)

    adjusted_evals = adjust(min.(1 ./ evals, 1), BenjaminiHochberg())
    rjs_idx = adjusted_evals .<= α
    (evalues=evals, rjs_idx=rjs_idx)
end 

function (::TTestEvalues)(Zs, α, effect_size; Ĝ=nothing, 𝒢=nothing, G_oracle=nothing, σs=nothing)

    Ts = Empirikos.NoncentralTSample.(Zs)

    mixture_evalue_T = Empirikos.MixtureEValue(Dirac(effect_size), Dirac(0.0))
    evals = mixture_evalue_T.(Ts)

    adjusted_evals = adjust(min.(1 ./ evals, 1), BenjaminiHochberg())
    rjs_idx = adjusted_evals .<= α
    (evalues=evals, rjs_idx=rjs_idx)
end 

function (::TTestPvalues)(Zs, α, effect_size; Ĝ=nothing, 𝒢=nothing, G_oracle=nothing, σs=nothing)
    sim_ttest = Empirikos.SimultaneousTTest(;α=α)
    sim_ttest_fit = fit(sim_ttest, Zs) 
    # not really evalues but just to keep it consistent
    (evalues=1 ./ sim_ttest_fit.pvalue, rjs_idx=sim_ttest_fit.rj_idx)
end 

function (::UniversalInferenceEvalues)(Zs, α, effect_size; Ĝ=nothing, 𝒢=nothing, G_oracle=nothing, σs=nothing)

    numerator_dbn = product_distribution((λ=Dirac(effect_size), σ²=Ĝ))

    universal_evalue = Empirikos.MixtureUniversalEValue(numerator_dbn)
    evals = universal_evalue.(Zs)

    adjusted_evals = adjust(min.(1 ./ evals, 1), BenjaminiHochberg())
    rjs_idx = adjusted_evals .<= α
    (evalues=evals, rjs_idx=rjs_idx)
end 

function (::FLocalizedMixtureEvalues)(Zs, α, effect_size; Ĝ=nothing, 𝒢=nothing, G_oracle=nothing, σs=nothing)

    evals_plugin = PluginMixtureEvalues()(Zs, α, effect_size; Ĝ=Ĝ, 𝒢=𝒢, G_oracle=G_oracle, σs=σs).evalues


    numerator_dbn = product_distribution((λ=Dirac(effect_size), σ²=Ĝ))

    eval_cui_numerators = pdf.(Ref(numerator_dbn), Zs)
    eval_cui_denominators = zero(eval_cui_numerators)
    evals = zero(eval_cui_numerators)

    # FLOC
    dkw = DvoretzkyKieferWolfowitz(α * 0.1 / exp(1)) #COMPOUND
    S² = ScaledChiSquareSample.(Zs)
    fitted_dkw = fit(dkw, S²)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    G = Empirikos.prior_variable!(model, 𝒢)
    Empirikos.flocalization_constraint!(model, fitted_dkw, G)
    #    @showprogress dt = 1 desc = "floc_iters" for i in eachindex(Zs)

    for i in eachindex(Zs)
        (evals_plugin[i] <=  1 / α) && continue
        Z_tmp = Zs[i]
        Msq = Empirikos.ScaledChiSquareSample(Z_tmp.mean_squares, Z_tmp.mean_squares_dof)
        @objective(model, Max,  pdf(G, Msq))
        optimize!(model)
        G_cui_var = G()
        eval_cui_denominators[i] = pdf(product_distribution((λ=Dirac(0), σ²=G_cui_var)), Z_tmp)
        evals[i] = eval_cui_numerators[i] / eval_cui_denominators[i]
    end

    adjusted_evals = adjust(min.(1 ./ evals, 1), BenjaminiHochberg())
    rjs_idx = adjusted_evals .<= α * 0.9
    (evalues=evals, rjs_idx=rjs_idx)
end 
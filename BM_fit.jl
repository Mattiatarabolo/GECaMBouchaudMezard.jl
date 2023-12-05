using Plots, GraphRecipes, StatsPlots
using GECaMBouchaudMezard
using Graphs
using Random
using JLD2
using BenchmarkTools


NV = 5000 # number of graph vertices
J = 1.0; # coupling

seed = 1234

dt = 0.01
t_init = 0.0
t_end = 2000.0;

Ks = [3, 10, 100, 1000]
σs = [round.(Vector(range(2*(K-1)-1.0, 2*(K-1)+1.0, length=10)), digits=2) for K in Ks]  # σ²_c ≈ 2(K-1)J from MF

dirpath_sol = "./data/single_sol/sol"
mkpath(dirpath_sol)
dirpath_fit = "./data/single_sol/fit"
mkpath(dirpath_save)
nbins = 100

for (iK, K) in enumerate(Ks)
    Threads.@threads for σ² in σs[iK]
        @load dirpath_sol*"/sol_N-$(N)_K$(K)_s2-$(σ²)_dt-$(dt)_T-$(t_end).jld" sol
        μ, pfit = pareto_fit(sol, idx_t(1800.0):idx_t(t_end), nbins, .85, .98)
        @save dirpath_fit*"/fit_N-$(N)_K$(K)_s2-$(σ²)_dt-$(dt)_T-$(t_end).jld" μ pfit
    end
end
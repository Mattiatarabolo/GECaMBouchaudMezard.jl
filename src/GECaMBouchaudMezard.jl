module GECaMBouchaudMezard

using Random
using SparseArrays, LinearAlgebra
using StatsBase, CurveFit
using Graphs
using Plots, StatsPlots
using JLD2

export BM_MilSDE, sim_BM_MilSDE, BM_MilSDE_JLD, sim_BM_MilSDE_JLD, idx_t, pareto_fit, pdf_norm_wealth, pareto_exponent

include("types.jl")
include("utils.jl")

include("simulation.jl")

end

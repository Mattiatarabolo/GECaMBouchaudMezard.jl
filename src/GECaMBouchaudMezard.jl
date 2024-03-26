module GECaMBouchaudMezard

using Random
using SparseArrays, LinearAlgebra
using StatsBase, CurveFit
using Graphs
using Plots, StatsPlots
using JLD2
using StaticArrays
using LoopVectorization
using LaTeXStrings
import ProgressMeter.@showprogress

export BM_MilSDE, BM_MilSDE_JLD, sim_BM_MilSDE, sim_BM_MilSDE_JLD, idx_t, rank_plot, hill_plot, improved_hill_plot, improved_hill_plot_2, pdf_norm_wealth

include("types.jl")
include("utils.jl")

include("simulation.jl")

end

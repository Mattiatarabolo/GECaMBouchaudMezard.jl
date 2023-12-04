using Plots, GraphRecipes, StatsPlots
using GECaMBouchaudMezard
using Graphs
using Random
using JLD2
using BenchmarkTools


NV = 5000 # number of graph vertices
J = 1.0; # coupling

seed = 1234

x_init = ones(NV)#rand(NV,1).*10 .+ 1

dt = 0.01
t_init = 0.0
t_end = 2000.0;

Ks = [3, 10, 100, 1000]
σs = [Vector(range(2*(K-1)-1.0, 2*(K-1)+1.0, length=10)) for K in Ks]  # σ²_c ≈ 2(K-1)J from MF

for K in Ks
    Threads.@threads for σ² in σs[K]
        # generate a random K-regualr graph with NV vertices
        G = random_regular_graph(NV, K)

        Amod = adjacency_matrix(G)

        for i in 1:NV
            Amod[i,i] = -Float64(K)
        end

        p = (J .* Amod, σ²)

        BM_MilSDE_JLD(p, dt, x_init, t_init, t_end, Threads.threadid())
    end

using Plots, GraphRecipes, StatsPlots
using GECaMBouchaudMezard
using Graphs
using Random
using JLD2
using BenchmarkTools
using QuadGK


NV = 5000 # number of graph vertices
σ² = 2.0 #fixed sigma

seed = 1234

x_init = ones(NV)#rand(NV,1).*10 .+ 1

dt = 0.01
t_init = 0.0
t_end = 2000.0;

Ks = [4, 10, 100, 1000]
Jcs = [0.0, 0.0, 0.0, 0.0]
for (iK, K) in enumerate(Ks)
    Jc, err = quadgk(x -> (2*σ²)/(pi*sqrt(K-1))*(sqrt(1-x^2))/(K/sqrt(K-1)-x+sqrt((K/sqrt(K-1)-x)^2-1)), -1, 1, rtol=1e-3)
    append!(Jcs, Jc)
end

Js = [round.(Vector(range(Jcs[iK]-1.0, Jcs[iK]+1.0, length=10)), digits=2) for (iK, K) in enumerate(Ks)] 

for (iK, K) in enumerate(Ks)

    # generate a random K-regualr graph with NV vertices
    G = random_regular_graph(NV, K)

    Amod = adjacency_matrix(G)

    for i in 1:NV
        Amod[i,i] = -Float64(K)
    end

    Threads.@threads for J in Js[iK]

        p = (J, Amod, σ²)

        println("Starting simulation sol_N-$(NV)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T$(t_end) on thread $(Threads.threadid())")
        BM_MilSDE_JLD(p, dt, x_init, t_init, t_end, Threads.threadid())
    end
end
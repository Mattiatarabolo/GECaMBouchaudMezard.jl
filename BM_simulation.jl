using Plots, GraphRecipes, StatsPlots
using GECaMBouchaudMezard
using Graphs
using Random
using JLD


NV = 300 # number of graph vertices
K = 3 # degree
σ² = 2.0 # noise amplitude
J = 1.0; # coupling

seed = 1234

x_init = ones(NV)#rand(NV,1).*10 .+ 1

dt = 0.01
t_init = 0.0
t_end = 500.0;


# generate a random K-regualr graph with NV vertices
G = random_regular_graph(NV, K)

Amod = adjacency_matrix(G)

for i in 1:NV
    Amod[i,i] = -Float64(K)
end

p = (J .* Amod, σ²);


nsim = 20

sim = sim_BM_MilSDE(p, dt, x_init, t_init, t_end, seed, nsim);


nbins = 100

μ, pfit = pareto_fit(sim, idx_t(1950, dt):idx_t(t_end, dt), nbins, .8, .99)

println(μ)

plot(pfit)
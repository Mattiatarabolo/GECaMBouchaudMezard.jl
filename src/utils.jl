function idx_t(t, dt)
    return floor(Int, t/dt)
end


function pdf_norm_wealth(sol::SDEsol, ts::UnitRange{Int64}, nbins::Int)
    wealths = Vector{Float64}()
    for t in ts
        append!(wealths, sol.xs[:, t])
    end
    filter!(x -> x>0, wealths)
    edge_l = log10(minimum(wealths))
    edge_r = log10(maximum(wealths))
    edges = 10.0 .^ (range(edge_l, edge_r, length=nbins))
    
    return StatsBase.normalize(StatsBase.fit(Histogram, wealths, edges), mode=:pdf)
end


function pdf_norm_wealth(sol::SDEsol, t::Int, nbins::Int)
    wealths = sol.xs[:, t]
    filter!(x -> x>0, wealths)
    edge_l = log10(minimum(wealths))
    edge_r = log10(maximum(wealths))
    edges = 10.0 .^ (range(edge_l, edge_r, length=nbins))
    
    return StatsBase.normalize(StatsBase.fit(Histogram, wealths, edges), mode=:pdf)
end


function pdf_norm_wealth(sim::SDEsim, ts::UnitRange{Int64}, nbins::Int)
    wealths = Vector{Float64}()
    for idx_sim in 1:sim.nsim
        for t in ts
            append!(wealths, sim.xs[idx_sim, :, t])
        end
    end
    filter!(x -> x>0, wealths)
    edge_l = log10(minimum(wealths))
    edge_r = log10(maximum(wealths))
    edges = 10.0 .^ (range(edge_l, edge_r, length=nbins))
    
    return StatsBase.normalize(StatsBase.fit(Histogram, wealths, edges), mode=:pdf)
end


function pdf_norm_wealth(sim::SDEsim, t::Int, nbins::Int)
    wealths = Vector{Float64}()
    for idx_sim in 1:sim.nsim
            append!(wealths, sim.xs[idx_sim, :, t])
    end
    filter!(x -> x>0, wealths)
    edge_l = log10(minimum(wealths))
    edge_r = log10(maximum(wealths))
    edges = 10.0 .^ (range(edge_l, edge_r, length=nbins))
    
    return StatsBase.normalize(StatsBase.fit(Histogram, wealths, edges), mode=:pdf)
end


function pareto_exponent(h::Histogram, frac_init::Float64, frac_end::Float64)
    nbins = length(h.edges[1])
    x = h.edges[1][2:end] .- (h.edges[1][2:end] .- h.edges[1][1:end-1]) ./ 2
    x = x[floor(Int, frac_init*nbins) - 1 : floor(Int, frac_end*nbins) - 1]
    y = h.weights
    y = y[floor(Int, frac_init*nbins) - 1 : floor(Int, frac_end*nbins) - 1]
    par_pow = power_fit(x, y)
    p = plot(h, xscale=:log10, yscale=:log10, fillalpha=.5, linewidth=0, linecolor=:match, c=:gray, xlims=(h.edges[1][1], h.edges[1][end]))
    xplot = range(x[1], x[end], length=100)
    plot!(xplot, par_pow[1] .* xplot .^ par_pow[2], width=4);
    return -par_pow[2], p
end


function pareto_fit(sol::SDEsol, t::Int, nbins::Int, frac_init::Float64, frac_end::Float64)
    h = pdf_norm_wealth(sol, t, nbins)
    return pareto_exponent(h, frac_init, frac_end)
end


function pareto_fit(sol::SDEsol, ts::UnitRange{Int64}, nbins::Int, frac_init::Float64, frac_end::Float64)
    h = pdf_norm_wealth(sol, ts, nbins)
    return pareto_exponent(h, frac_init, frac_end)
end


function pareto_fit(sim::SDEsim, t::Int, nbins::Int, frac_init::Float64, frac_end::Float64)
    h = pdf_norm_wealth(sim, t, nbins)
    return pareto_exponent(h, frac_init, frac_end)
end


function pareto_fit(sim::SDEsim, ts::UnitRange{Int64}, nbins::Int, frac_init::Float64, frac_end::Float64)
    h = pdf_norm_wealth(sim, ts, nbins)
    return pareto_exponent(h, frac_init, frac_end)
end


function save_JLD(sol, p, dt, t_end)
    K = Int(p[1][1,1])
    N = size(p[1])[1]
    σ² = p[2]

    jldopen("./data/single_sol/sol_N$(N)_K$(K)_s2$(σ²)_dt$(dt)_T$(t_end).jld", "w") do f
        write(f, "sol", sol)
    end
end


function save_JLD(sol, p, dt, t_end, i, thread_id)
    K = Int(p[1][1,1])
    N = size(p[1])[1]
    σ² = p[2]

    dirpath = "./data/sim/N-$(N)_K-$(K)_s2-$(σ²)_dt-$(dt)_T-$(t_end)"

    mkpath(dirpath)

    @save dirpath*"/sol_N-$(N)_K-$(K)_s2-$(σ²)_dt-$(dt)_T-$(t_end)_$(i).jld" sol

    println("writing sol_N-$(N)_K-$(K)_s2-$(σ²)_dt-$(dt)_T-$(t_end)_$(i) on thread $(thread_id)")
end
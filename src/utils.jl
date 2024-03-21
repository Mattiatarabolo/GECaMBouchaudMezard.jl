function idx_t(t, dt)
    return floor(Int, t/dt) + 1
end

function Y2(xs_sim::Array{Float64, 3})
    return mean(sum(xs_sim, dims=2), dims=(1,3))[1]
end

function save_JLD(xs::Matrix{Float64}, N::Int, K::Int, J::Float64, σ²::Float64, dt::Float64, t_end::Float64)

    dirpath = "./data/single_sol/sol"
    mkpath(dirpath)

    @save dirpath*"/sol_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-$(t_end).jld" xs
 end


function save_JLD(xs::Matrix{Float64}, N::Int, K::Int, J::Float64, σ²::Float64, dt::Float64, t_end::Float64, idx_sim::Int, dirpath::String)

    @save dirpath*"/sol_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-$(t_end)_$(idx_sim).jld" xs

end


function save_JLD(xs_sim::Array{Float64, 3}, N::Int, K::Int, J::Float64, σ²::Float64, dt::Float64, t_end::Float64)

    dirpath = "./data/sim/sim_arrays"

    mkpath(dirpath)

    @save dirpath*"/sim_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-$(t_end).jld" xs_sim

    println("writing sim_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-$(t_end)")
end


function rankplot(x::Vector{Float64})
    x_ranked = sort(x, rev=true)
    p_rank = scatter(x_ranked, xscale=:log2, yscale=:log2, minorgrid=true, xlabel="rank", ylabel=L"w", markershape=:diamond)
    return p_rank, x_ranked
end

function hill_plot(x_ranked_ln::Vector{Float64}, k_max::Int)
    α_hill = ones(k_max)
    for k in 1:k_max
        α_hill[k] /= mean(x_ranked_ln[1:k]) - x_ranked_ln[k+1]
    end

    p_hill = scatter(1:k_max, α_hill, xscale=:ln, xlabel="rank", ylabel=L"\hat{\alpha}(k)", markershape=:diamond)
    return p_hill, α_hill    
end

function improved_hill_plot(x_ranked_ln::Vector{Float64}, k_max::Int, j_max::Int)
    if j_max >= k_max
        j_max = k_max - 2
    end        
    α_hill = [ones(k_max-j) for j in 0:j_max]
    ks = [j+1:k_max for j in 0:j_max]
    @showprogress for (ji, j) in enumerate(0:j_max)
        for (ki, k) in enumerate(ks[ji])
            α_hill[ji][ki] /= j/(k-j)*x_ranked_ln[j+1]-k/(k-j)*x_ranked_ln[k+1]+mean(x_ranked_ln[j+1:k])
        end
    end
    return α_hill    
end
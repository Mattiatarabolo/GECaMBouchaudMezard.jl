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
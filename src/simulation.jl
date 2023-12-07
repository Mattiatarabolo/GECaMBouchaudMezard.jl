function Wiener_diag!(ΔW::Vector{Float64}, N::Int)
    @turbo for i in 1:N
        ΔW[i] = randn()
    end
end

f!(dx::Vector{Float64}, x::Vector{Float64}, p::Tuple{Float64, SparseMatrixCSC, Float64}) = mul!(dx, p[1].*p[2], x)
g!(dx::Vector{Float64}, x::Vector{Float64}, p::Tuple{Float64, SparseMatrixCSC, Float64}) =  mul!(dx, sqrt(p[3]), x)
g_Mil!(dx::Vector{Float64}, x::Vector{Float64}, p::Tuple{Float64, SparseMatrixCSC, Float64}) = mul!(dx, p[3]/2, x)

function Mil_update!(x::Vector{Float64}, Δx_det::Vector{Float64}, Δx_stoch::Vector{Float64}, Δx_Mil::Vector{Float64}, ΔW::Vector{Float64}, dt::Float64, N::Int)
    @turbo for i in 1:N
        x[i] = x[i] + Δx_det[i]*dt + Δx_stoch[i]*ΔW[i]*sqrt(dt) + Δx_Mil[i]*(ΔW[i]^2 - 1)*dt
    end
end

function sol_update!(xs::Matrix{Float64}, x::Vector{Float64}, τ::Int, N::Int, av::Float64)
    av = mean(x)
    @turbo for i in 1:N
        xs[i, τ] = x[i]/av
    end
end

function mat_update!(xs::Matrix{Float64}, x::Vector{Float64}, τ::Int, N::Int)
    @turbo for i in 1:N
        xs[i, τ] = x[i]
    end
end

function BM_MilSDE(p::Tuple{Float64, SparseMatrixCSC, Float64}, dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, N::Int)
    T = floor(Int, (t_end-t_init)/dt) + 1
    
    xs = zeros(N, T)

    mat_update!(xs, x_init, 1, N)
    x_mean = mean

    x =  x_init
    Δx_det = zeros(N)
    Δx_stoch = zeros(N)
    Δx_Mil = zeros(N)
    ΔW = zeros(N)

    # integration loop
    @showprogress for τ in 2:T
        f!(Δx_det, x, p)
        g!(Δx_stoch, x, p)
        g_Mil!(Δx_Mil, x, p)
        Wiener_diag!(ΔW, N)

        # Milstein update

        Mil_update!(x, Δx_det, Δx_stoch, Δx_Mil, ΔW, dt, N)
        sol_update!(xs, x, τ, N, x_mean)
    end
    return xs
end


function BM_MilSDE_JLD(p::Tuple{Float64, SparseMatrixCSC, Float64}, dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, N::Int, thread_id::Int)
    println("Starting sol_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-(t_end)_$(idx_sim) on thread $(Threads.threadid())")
    
    T = floor(Int, (t_end-t_init)/dt) + 1
    
    xs = zeros(N, T)

    mat_update!(xs, x_init, 1, N)
    x_mean = mean

    x =  x_init
    Δx_det = zeros(N)
    Δx_stoch = zeros(N)
    Δx_Mil = zeros(N)
    ΔW = zeros(N)

    # integration loop
    @showprogress for τ in 2:T
        f!(Δx_det, x, p)
        g!(Δx_stoch, x, p)
        g_Mil!(Δx_Mil, x, p)
        Wiener_diag!(ΔW, N)

        # Milstein update

        Mil_update!(x, Δx_det, Δx_stoch, Δx_Mil, ΔW, dt, N)
        sol_update!(xs, x, τ, N, x_mean)
    end

    save_JLD(xs, p, dt, t_end, thread_id)
end


function BM_MilSDE_JLD(p::Tuple{Float64, SparseMatrixCSC, Float64}, dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, N::Int)
    T = floor(Int, (t_end-t_init)/dt) + 1
    
    xs = zeros(N, T)

    mat_update!(xs, x_init, 1, N)
    x_mean = mean

    x =  x_init
    Δx_det = zeros(N)
    Δx_stoch = zeros(N)
    Δx_Mil = zeros(N)
    ΔW = zeros(N)

    # integration loop
    for τ in 2:T
        f!(Δx_det, x, p)
        g!(Δx_stoch, x, p)
        g_Mil!(Δx_Mil, x, p)
        Wiener_diag!(ΔW, N)

        # Milstein update

        Mil_update!(x, Δx_det, Δx_stoch, Δx_Mil, ΔW, dt, N)
        sol_update!(xs, x, τ, N, x_mean)
    end

    save_JLD(xs, p, dt, t_end)
end


###################################### SIMULATIONS ########################################################
function mat_update!(xs_sim::Array{Float64, 3}, xs::Matrix{Float64}, idx_sim::Int, N::Int, T::Int)
    @turbo for i in 1:N
        for τ in 1:T
            xs_sim[idx_sim, i, τ] = xs[i, τ]
        end
    end
end

function sim_BM_MilSDE(dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, N::Int, K::Int, σ²::Float64, J::Float64, seed::Int, nsim::Int)
    T = floor(Int, (t_end-t_init)/dt) + 1

    xs_sim = zeros(nsim, N, T)

    Threads.@threads for idx_sim in 1:nsim
        G = random_regular_graph(N, K)
        Amod = adjacency_matrix(G)
        for i in 1:NV
            Amod[i,i] = -Float64(K)
        end

        p = (J, Amod, σ²)

        Random.seed!(seed)
        xs = BM_MilSDE(p, dt, x_init, t_init, t_end, N)

        mat_update!(xs_sim, xs, idx_sim, N, T)
    end

    return xs_sim
end


function sim_BM_MilSDE_JLD(dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, N::Int, K::Int, σ²::Float64, J::Float64, seed::Int, nsim::Int)
    T = floor(Int, (t_end-t_init)/dt) + 1
    
    xs_sim = zeros(nsim, N, T)

    Threads.@threads for idx_sim in 1:nsim
        G = random_regular_graph(N, K)
        Amod = adjacency_matrix(G)
        for i in 1:NV
            Amod[i,i] = -Float64(K)
        end

        p = (J, Amod, σ²)

        println("Starting sol_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-(t_end)_$(idx_sim) on thread $(Threads.threadid())")
        Random.seed!(seed)
        xs = BM_MilSDE(p, dt, x_init, t_init, t_end, N)
        
        mat_update!(xs_sim, xs, idx_sim, N, T)
        println("Completed sol_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-(t_end)_$(idx_sim) on thread $(Threads.threadid())")
    end
    save_JLD(xs_sim, p, dt, t_end)
end


function sim_BM_MilSDE(p::Tuple{Float64, SparseMatrixCSC, Float64}, dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, N::Int, seed::Int, nsim::Int)
    T = floor(Int, (t_end-t_init)/dt) + 1

    xs_sim = zeros(nsim, N, T)

    Threads.@threads for idx_sim in 1:nsim

        Random.seed!(seed + idx_sim)
        xs = BM_MilSDE(p, dt, x_init, t_init, t_end, N)

        mat_update!(xs_sim, xs, idx_sim, N, T)
    end

    return sim
end


function sim_BM_MilSDE_JLD(p::Tuple{Float64, SparseMatrixCSC, Float64}, dt::Float64, x_init::Vector{Float64}, t_init::Float64, t_end::Float64, N::Int, seed::Int, nsim::Int)
    T = floor(Int, (t_end-t_init)/dt) + 1

    xs_sim = zeros(nsim, N, T)

    Threads.@threads for idx_sim in 1:nsim
        println("Starting sol_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-(t_end)_$(idx_sim) on thread $(Threads.threadid())")
        Random.seed!(seed + idx_sim)
        xs = BM_MilSDE(p, dt, x_init, t_init, t_end, N)
        println("Completed sol_N-$(N)_K$(K)_J-$(J)_s2-$(σ²)_dt-$(dt)_T-(t_end)_$(idx_sim) on thread $(Threads.threadid())")
    end
    save_JLD(xs_sim, p, dt, t_end)
end
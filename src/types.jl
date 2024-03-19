struct SDEsol
    t::Vector{Float64}
    xs::Array{Float64, 2}
    dt::Float64
    par::Tuple{Float64, SparseMatrixCSC, Float64}

    function SDEsol(t::Vector{Float64}, N::Int, dt::Float64, par::Tuple{Float64, SparseMatrixCSC, Float64})
        new(t, zeros(N, length(t)), dt, par)
    end
end




struct SDEsim
    t::Vector{Float64}
    xs::Array{Float64, 3}
    dt::Float64
    pars::Vector{Tuple{Float64, SparseMatrixCSC, Float64}}
    nsim::Int

    function SDEsim(t::Vector{Float64}, N::Int, dt::Float64, nsim::Int)
        new(t, zeros(nsim, N, length(t)), dt, [(0.0, zeros(N, N), 0.0) for _ in 1:nsim], nsim)
    end
end




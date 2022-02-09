# CG preconditioners must be symmetric and positive definite.
# Multigrid cycles will satisfy these requirements for V- and W-cycles if
#   * Restriction is the adjoint of prolongation (up to scaling)
#   * Postsmoothing is performed in reverse order of presmoothing
#   * The coarse solver is either exact or symmetric and positive definite

__DEFAULT_MGPCG_DEPTH(A) = max(2, 1 + floor(Int, log2(minimum(size(A)))-2))

__DEFAULT_MGPCG_PRE(nlevels::Integer) = ntuple(nlevels-1) do l
    Smoothers(
        RedBlackGaussSeidel(ForwardSweep(), iter=1),
        RedBlackGaussSeidel(BoundarySweep(ForwardSweep()), iter=2^l),
    )
end

__DEFAULT_MGPCG_COARSE() = Smoothers(
    RedBlackGaussSeidel(ForwardSweep(), iter=512),
    RedBlackGaussSeidel(BackwardSweep(), iter=512),
)

__DEFAULT_MGPCG_POST(nlevels::Integer) = ntuple(nlevels-1) do l
    Smoothers(
        RedBlackGaussSeidel(BoundarySweep(BackwardSweep()), iter=2^l),
        RedBlackGaussSeidel(BackwardSweep(), iter=1),
    )
end


function mgpcg(
    A::Poisson2{<:AbstractFloat, 3},
    b::AbstractArray{<:AbstractFloat, N};
    kwargs...
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return mgpcg!(tzero(b), A, b; kwargs...)
end

function mgpcg!(
    x::AbstractArray{<:AbstractFloat, N},
    A::Poisson2{<:AbstractFloat, 3},
    b::AbstractArray{T};
    maxlevel::Integer = __DEFAULT_MGPCG_DEPTH(A),
    presmoother = __DEFAULT_MGPCG_PRE(maxlevel),
    coarsesolver = __DEFAULT_MGPCG_COARSE(),
    postsmoother = __DEFAULT_MGPCG_POST(maxlevel),
    cycle::Cycle = V(),
    ncycles::Integer = 1,
    atol::Real = zero(T),
    rtol::Real = sqrt(eps(T)),
    maxit::Integer = floor(Int, sqrt(length(b))),
    verbose::Bool = false
) where {T<:AbstractFloat, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    size(x) == size(b) || throw(DimensionMismatch())
    size(A) == size(b)[1:3] || throw(DimensionMismatch())

    M = Multigrid(T, A,
        coarsesolver = coarsesolver,
        presmoother = presmoother,
        postsmoother = postsmoother,
        maxlevel = maxlevel
    )

    q = tzero(b, size(A))
    p = similar(b, size(A))

    M.workspace.x[1] = q

    if N == 3
        M.workspace.b[1] = tcopy(b)
        x = _mgpcg!(x, M, p, cycle, ncycles, atol, rtol, maxit, verbose)

    elseif N == 4
        xt = similar(x, size(A))
        M.workspace.b[1] = similar(b, size(A))

        for t in axes(b, 4)
            _tcopyto!(xt, @view(x[:,:,:,t]))
            _tcopyto!(M.workspace.b[1], @view(b[:,:,:,t]))

            xt = _mgpcg!(xt, M, p, cycle, ncycles, atol, rtol, maxit, verbose)

            _tcopyto!(@view(x[:,:,:,t]), xt)
        end
    end

    return x
end

function _mgpcg!(
    x::AbstractArray{<:AbstractFloat, 3},
    M::Multigrid,
    p::AbstractArray{T, 3},
    cycle::Cycle,
    ncycles::Integer,
    atol::Real,
    rtol::Real,
    maxit::Integer,
    verbose::Bool
) where {T<:AbstractFloat}
    _zero = zero(T)
    ρ = one(T)

    A = M.grids[1].A
    q = M.workspace.x[1]
    r = M.workspace.b[1]
    p = tfill!(p, _zero)

    # q = A * x
    # r -= q
    # res = norm(r)
    r, q, res = _init_r!(r, q, A, x)

    tol = max(rtol * res, atol)

    if verbose
        @printf("==== MGPCG ====\n")
        @printf("Depth: %d\n", length(M.grids))
        @printf("Tolerance: %1.4e\n", tol)
        @printf("%8s %3s %9s\n", "iter", " ", "resnorm")
    end

    if res ≤ tol
        return x
    end

    for i in 1:maxit
        # q = M\r
        tfill!(q, _zero, A)
        for _ in 1:ncycles
            cycle!(M, cycle)
        end

        # ρ = q⋅r
        ρ_prev = ρ
        ρ = _compute_ρ(q, r, A)

        # p = q + β * p
        β = ρ / ρ_prev
        p = _compute_p!(p, q, β, A)

        # q = Ap
        # α = q⋅r / p⋅Ap
        q, pAp = _compute_q!(q, A, p)
        α = ρ / pAp

        # x = x + α * p
        # r = r - α * q
        # res = norm(r)
        x, r, res = _compute_x_r!(x, r, p, q, α, A)

        if verbose
            @printf("[%5d/%d]  %1.4e\n", i, maxit, res)
        end

        if res ≤ tol
            break
        end
    end

    return x
end


function _compute_ρ(q::AbstractArray{T}, r, A::Poisson2) where {T}
    # ρ = q⋅r
    @inbounds @batch threadlocal=zero(T)::T for I in A.RI
        threadlocal = muladd(q[I], r[I], threadlocal)
    end
    return sum(threadlocal::Vector{T})
end


function _compute_p!(p, q, β, A::Poisson2)
    # p = q + β * p
    @inbounds @batch for I in A.RI
        p[I] *= β
        p[I] += q[I]
    end
    return p
end


function _compute_q!(q::AbstractArray{T}, A::Poisson2, p) where {T}
    # q = A * p
    # α = p*A*p
    D = A.D
    idx2 = convert(T, A.idx2[1])
    idy2 = convert(T, A.idx2[2])
    idz2 = convert(T, A.idx2[3])

    @inbounds @batch per=thread threadlocal=zero(T)::T for (I, J, K) in A.R8
        for k in K, j in J, i in I
            z1 = p[i,j,k-1]
            y1 = p[i,j-1,k]
            x1 = p[i-1,j,k]
            x0 = p[i,j,k]
            x2 = p[i+1,j,k]
            y2 = p[i,j+1,k]
            z2 = p[i,j,k+1]

            if A.interior[i,j,k]
                Δz = z1 + z2
                Δy = y1 + y2
                Δx = x1 + x2

                Δ = D * x0
                Δ = muladd(idz2, Δz, Δ)
                Δ = muladd(idy2, Δy, Δ)
                Δ = muladd(idx2, Δx, Δ)

                q[i,j,k] = Δ
                threadlocal = muladd(x0, Δ, threadlocal)
            end
        end
    end

    return q, sum(threadlocal::Vector{T})
end


function _compute_x_r!(x::AbstractArray{T}, r, p, q, α, A::Poisson2) where {T}
    # x = x + α * p
    # r = r - α * q
    # res = norm(r)
    @inbounds @batch threadlocal=zero(T)::T for I in A.RI
        x[I] = muladd( α, p[I], x[I])
        r[I] = muladd(-α, q[I], r[I])
        threadlocal = muladd(r[I], r[I], threadlocal)
    end
    return x, r, sqrt(sum(threadlocal::Vector{T}))
end


function _init_r!(r::AbstractArray{T}, q, A, x) where {T}
    # q = A * x
    # r -= q
    # res = norm(r)
    q = mul!(q, A, x)
    @inbounds @batch threadlocal=zero(T)::T for I in eachindex(r)
        r[I] -= q[I]
        threadlocal = muladd(r[I], r[I], threadlocal)
    end
    return r, q, sqrt(sum(threadlocal::Vector{T}))
end

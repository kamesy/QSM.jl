abstract type Cycle end
struct V <: Cycle end
struct W <: Cycle end
struct F <: Cycle end


struct Grid{TA, Pre, Post}
    A::TA
    presmoother::Pre
    postsmoother::Post
end

Base.size(grid::Grid) = size(grid.A)
Base.eltype(grid::Grid) = eltype(grid.A)


struct MultigridWorkspace{Tx}
    x::Vector{Tx}
    b::Vector{Tx}
    r::Vector{Tx}
end


struct Multigrid{TA, Tx, Coarse}
    grids::Vector{TA}
    coarsesolver::Coarse
    workspace::MultigridWorkspace{Tx}
end

Multigrid(A::Poisson2{T}; kwargs...) where {T} = Multigrid(T, A; kwargs...)

function Multigrid(
    T,
    A::Poisson2;
    presmoother = RedBlackGaussSeidel(ForwardSweep(), iter=2),
    coarsesolver = EarlyStopping(RedBlackGaussSeidel(ForwardSweep(), iter=100), sqrt(eps(T)), 1000),
    postsmoother = RedBlackGaussSeidel(ForwardSweep(), iter=2),
    maxlevel = max(2, floor(Int, log2(minimum(size(A)))-2)),
)
    maxlevel > 1 || throw(ArgumentError(""))

    sz = size(A)
    Tx = Array{T}

    presmoother = expand(presmoother, maxlevel)
    postsmoother = expand(postsmoother, maxlevel)
    grids = Vector{Grid}([Grid(A, presmoother[1], postsmoother[1])])

    x = Tx(undef, sz .* 0) # placeholder
    b = Tx(undef, sz .* 0) # placeholder
    r = tfill!(Tx(undef, sz), zero(T))
    workspace = MultigridWorkspace([x], [b], [r])

    for l in 2:maxlevel
        Ac = restrict(grids[l-1].A)
        push!(grids, Grid(Ac, presmoother[l], postsmoother[l]))

        szc = size(Ac)
        x = tfill!(Tx(undef, szc), zero(T))
        b = tfill!(Tx(undef, szc), zero(T))
        r = tfill!(Tx(undef, szc), zero(T))
        push!(workspace.x, x)
        push!(workspace.b, b)
        push!(workspace.r, r)
    end

    return Multigrid(grids, coarsesolver, workspace)
end


function solve(M::Multigrid, b::AbstractArray, args...; kwargs...)
    return solve!(tzero(b), M, b, args...; kwargs...)
end


function solve!(
    x::AbstractArray{<:AbstractFloat, N},
    M::Multigrid,
    b::AbstractArray{T, N},
    cycle::Cycle = V();
    atol::Real = zero(T),
    rtol::Real = sqrt(eps(T)),
    maxit::Integer = 100,
    residual::Bool = true,
    log::Bool = false,
    verbose::Bool = false,
) where {T<:AbstractFloat, N}
    size(x) == size(b) || throw(DimensionMismatch())
    size(x) == size(M.grids[1].A) || throw(DimensionMismatch())

    M.workspace.x[1] = x
    M.workspace.b[1] = b

    A = M.grids[1].A

    if residual
        nr = norm(b)
        ϵ = max(rtol * nr, atol)

        if log
            ch = [nr]
        end

        if verbose
            @printf("==== Multigrid ====\n")
            @printf("Depth: %d\n", length(M.grids))
            @printf("Tolerance: %1.4e\n", ϵ)
            @printf("%6s %3s %9s %7s\n", "iter", " ", "resnorm", "ratio")
            @printf("[%3d/%3d]  %1.4e\n", 0, maxit, nr)
        end
    end

    for i in 1:maxit
        cycle!(M, cycle)

        if residual
            nr0 = nr
            nr = __norm_residual(b, A, x)

            if log
                push!(ch, nr)
            end

            if verbose
                @printf("[%3d/%3d]  %1.4e  %0.2f\n", i, maxit, nr, nr / nr0)
            end

            if residual && nr ≤ ϵ
                break
            end
        end
    end

    if residual && verbose
        println()
    end

    return log ? (x, ch) : x
end


@noinline function cycle!(M::Multigrid, cycle::Cycle, l::Integer = 1)
    A = M.grids[l].A
    Ac = M.grids[l+1].A

    presmooth! = M.grids[l].presmoother
    postsmooth! = M.grids[l].postsmoother

    x = M.workspace.x[l]
    b = M.workspace.b[l]
    r = M.workspace.r[l]

    xc = M.workspace.x[l+1]
    bc = M.workspace.b[l+1]

    presmooth!(x, A, b)
    residual!(r, b, A, x)
    restrict!(bc, Ac, r)

    xc = tfill!(xc, 0, Ac)

    if l+1 == length(M.grids)
        M.coarsesolver(xc, Ac, bc)
    else
        _cycle!(M, cycle, l+1)
    end

    correct_prolong!(x, A, xc)
    postsmooth!(x, A, b)

    return M
end


function _cycle!(M, cycle::V, l)
    cycle!(M, cycle, l)
end

function _cycle!(M, cycle::W, l)
    cycle!(M, cycle, l)
    cycle!(M, cycle, l)
end

function _cycle!(M, cycle::F, l)
    cycle!(M, cycle, l)
    cycle!(M, V(), l)
end

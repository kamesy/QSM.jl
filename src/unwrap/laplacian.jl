"""
    unwrap_laplacian(
        phas::AbstractArray{<:AbstractFloat, N ∈ (3, 4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        solver::Symbol = :mgpcg
    ) -> typeof(similar(phas))

Laplacian phase unwrapping [1].

The Laplacian is computed using second order central finite differences
on the complex phase.
The resulting Poisson's equation can be solved with homogeneous Dirichlet
(`solver = :mgpcg`), Neumann (`:dct`), or periodic (`:fft`) boundary
conditions (BCs). Neumann and periodic BCs are imposed on the array while
Dirichlet BCs are imposed on a region-of-interest (ROI) (`mask`). The boundary
of the ROI is set such that values outside of it (`mask = 0`) are taken as
boundary points and values inside of it (`mask = 1`) as interior points, ie.
BC: `uphas[!mask] = 0`. This method combines phase unwrapping [1] and
harmonic background field removing [2].

### Arguments
- `phas::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: wrapped (multi-echo) phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size

### Keywords
- `solver::Symbol = :mgpcg`: solver for Poisson equation
    - `:dct`: homogeneous Neumann boundary condition on array
    - `:fft`: periodic boundary condition on array
    - `:mgpcg`: homogeneous Dirichlet boundary condition on `mask`
        (multigrid-preconditioned conjugate gradient method)

### Returns
- `typeof(similar(phas))`: unwrapped phase

### References
[1] Schofield MA, Zhu Y. Fast phase unwrapping algorithm for interferometric
    applications. Optics letters. 2003 Jul 15;28(14):1194-6.

[2] Zhou D, Liu T, Spincemaille P, Wang Y. Background field removal by solving
    the Laplacian boundary value problem. NMR in Biomedicine. 2014 Mar;27(3):312-9.
"""
function unwrap_laplacian(
    phas::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    solver::Symbol = :mgpcg
) where {T<:AbstractFloat, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    checkshape(axes(mask), axes(phas)[1:3], (:mask, :phas))

    solver ∈ (:dct, :fft, :mgpcg) ||
        throw(ArgumentError("solver must be one of :dct, :fft, :mgpcg"))

    d2uphas = wrapped_laplacian(phas, vsz)

    if solver == :dct
        d2uphas = wrapped_laplacian_boundary_neumann!(d2uphas, phas, vsz)
        uphas = solve_poisson_dct(d2uphas, vsz)

    elseif solver == :fft
        d2uphas = wrapped_laplacian_boundary_periodic!(d2uphas, phas, vsz)
        uphas = solve_poisson_fft(d2uphas, vsz)

    elseif solver == :mgpcg
        # src/utils/poisson_solver/mgpcg.jl
        nlevels = __DEFAULT_MGPCG_DEPTH(mask)

        opts = (
            # src/bgremove/lbv.jl
            presmoother  = __DEFAULT_LBV_PRE(nlevels),
            coarsesolver = __DEFAULT_LBV_COARSE(),
            postsmoother = __DEFAULT_LBV_POST(nlevels),
            atol = sqrt(eps(Float64)),
            rtol = sqrt(eps(Float64)),
            maxit = maximum(size(mask)),
        )

        # set boundaries
        @inbounds for t in axes(d2uphas, 4)
            d2ut = @view(d2uphas[:,:,:,t])
            @batch for I in eachindex(d2ut, mask)
                d2ut[I] *= mask[I]
            end
        end

        uphas = solve_poisson_mgpcg(d2uphas, mask, vsz; opts...)
    end

    return uphas
end


function wrapped_laplacian(
    u::AbstractArray{<:AbstractFloat, N},
    dx::NTuple{3, Real}
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return wrapped_laplacian!(tzero(u), u, dx)
end

function wrapped_laplacian!(
    d2u::AbstractArray{<:AbstractFloat, 3},
    u::AbstractArray{T, 3},
    dx::NTuple{3, Real}
) where {T<:AbstractFloat}
    checkshape(d2u, u, (:d2u, :u))

    nx, ny, nz = size(u)
    dx2, dy2, dz2 = convert.(T, inv.(dx.*dx))

    tsz = padded_tilesize(T, (2, 2, 2), 1)
    R = vec(collect(TileIterator((2:nx-1, 2:ny-1, 2:nz-1), tsz)))

    @inbounds @batch per=thread for (I, J, K) in R
        for k in K
            for j in J
                for i in I
                    Δ =        -dz2 *_wrap(u[i,j,k] - u[i,j,k-1])
                    Δ = muladd(-dy2, _wrap(u[i,j,k] - u[i,j-1,k]), Δ)
                    Δ = muladd(-dx2, _wrap(u[i,j,k] - u[i-1,j,k]), Δ)
                    Δ = muladd( dx2, _wrap(u[i+1,j,k] - u[i,j,k]), Δ)
                    Δ = muladd( dy2, _wrap(u[i,j+1,k] - u[i,j,k]), Δ)
                    Δ = muladd( dz2, _wrap(u[i,j,k+1] - u[i,j,k]), Δ)
                    d2u[i,j,k] = Δ
                end
            end
        end
    end

    return d2u
end

function wrapped_laplacian!(
    d2u::AbstractArray{<:AbstractFloat, 4},
    u::AbstractArray{T, 4},
    dx::NTuple{3, Real}
) where {T<:AbstractFloat}
    checkshape(d2u, u, (:d2u, :u))

    nx, ny, nz, _ = size(u)
    dx2, dy2, dz2 = convert.(T, inv.(dx.*dx))

    tsz = padded_tilesize(T, (2, 2, 2), 1)
    R = vec(collect(TileIterator((2:nx-1, 2:ny-1, 2:nz-1), tsz)))

    @inbounds for t in axes(u, 4)
        _u = @view(u[:,:,:,t])
        _d2u = @view(d2u[:,:,:,t])
        @batch per=thread for (I, J, K) in R
            for k in K
                for j in J
                    for i in I
                        Δ =        -dz2 *_wrap(_u[i,j,k] - _u[i,j,k-1])
                        Δ = muladd(-dy2, _wrap(_u[i,j,k] - _u[i,j-1,k]), Δ)
                        Δ = muladd(-dx2, _wrap(_u[i,j,k] - _u[i-1,j,k]), Δ)
                        Δ = muladd( dx2, _wrap(_u[i+1,j,k] - _u[i,j,k]), Δ)
                        Δ = muladd( dy2, _wrap(_u[i,j+1,k] - _u[i,j,k]), Δ)
                        Δ = muladd( dz2, _wrap(_u[i,j,k+1] - _u[i,j,k]), Δ)
                        _d2u[i,j,k] = Δ
                    end
                end
            end
        end
    end

    return d2u
end


#####
##### Boundary treatment
#####

# unroll? probably too aggressive
function wrapped_laplacian_boundary_neumann!(
    d2u::AbstractArray{<:AbstractFloat, N},
    u::AbstractArray{T, N},
    dx::NTuple{3, Real}
) where {T<:AbstractFloat, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    checkshape(d2u, u, (:d2u, :u))

    sz = size(u)
    nx, ny, nz = sz[1:3]
    dx2, dy2, dz2 = convert.(T, inv.(dx.*dx))

    _zero = zero(T)

    outer = CartesianIndices(ntuple(n -> 1:sz[n], Val(3)))
    inner = CartesianIndices(ntuple(n -> 2:sz[n]-1, Val(3)))
    E = EdgeIterator(outer, inner)

    R = Vector{NTuple{3, Int}}(undef, length(E))
    @inbounds for (i, I) in enumerate(E)
        R[i] = Tuple(I)
    end

    @inbounds for t in axes(u, 4)
        _u = @view(u[:,:,:,t])
        _d2u = @view(d2u[:,:,:,t])
        @batch per=thread for (i, j, k) in R
            Δ = _zero

            if k > 1
                Δ = muladd(-dz2, _wrap(_u[i,j,k] - _u[i,j,k-1]), Δ)
            end

            if j > 1
                Δ = muladd(-dy2, _wrap(_u[i,j,k] - _u[i,j-1,k]), Δ)
            end

            if i > 1
                Δ = muladd(-dx2, _wrap(_u[i,j,k] - _u[i-1,j,k]), Δ)
            end

            if i < nx
                Δ = muladd( dx2, _wrap(_u[i+1,j,k] - _u[i,j,k]), Δ)
            end

            if j < ny
                Δ = muladd( dy2, _wrap(_u[i,j+1,k] - _u[i,j,k]), Δ)
            end

            if k < nz
                Δ = muladd( dz2, _wrap(_u[i,j,k+1] - _u[i,j,k]), Δ)
            end

            _d2u[i,j,k] = Δ
        end
    end

    return d2u
end


function wrapped_laplacian_boundary_periodic!(
    d2u::AbstractArray{<:AbstractFloat, N},
    u::AbstractArray{T, N},
    dx::NTuple{3, Real}
) where {T<:AbstractFloat, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    checkshape(d2u, u, (:d2u, :u))

    sz = size(u)
    nx, ny, nz = sz[1:3]
    dx2, dy2, dz2 = convert.(T, inv.(dx.*dx))

    outer = CartesianIndices(ntuple(n -> 1:sz[n], Val(3)))
    inner = CartesianIndices(ntuple(n -> 2:sz[n]-1, Val(3)))
    E = EdgeIterator(outer, inner)

    R = Vector{NTuple{3, Int}}(undef, length(E))
    @inbounds for (i, I) in enumerate(E)
        R[i] = Tuple(I)
    end

    @inbounds for t in axes(u, 4)
        _u = @view(u[:,:,:,t])
        _d2u = @view(d2u[:,:,:,t])
        @batch per=thread for (i, j, k) in R
            du = k == 1 ? _u[i,j,k] - _u[i,j,end] : _u[i,j,k] - _u[i,j,k-1]
            Δ = -dz2 * _wrap(du)

            du = j == 1 ? _u[i,j,k] - _u[i,end,k] : _u[i,j,k] - _u[i,j-1,k]
            Δ = muladd(-dy2, _wrap(du), Δ)

            du = i == 1 ? _u[i,j,k] - _u[end,j,k] : _u[i,j,k] - _u[i-1,j,k]
            Δ = muladd(-dx2, _wrap(du), Δ)

            du = i == nx ? _u[1,j,k] - _u[i,j,k] : _u[i+1,j,k] - _u[i,j,k]
            Δ = muladd(dx2, _wrap(du), Δ)

            du = j == ny ? _u[i,1,k] - _u[i,j,k] : _u[i,j+1,k] - _u[i,j,k]
            Δ = muladd(dy2, _wrap(du), Δ)

            du = k == nz ? _u[i,j,1] - _u[i,j,k] : _u[i,j,k+1] - _u[i,j,k]
            Δ = muladd(dz2, _wrap(du), Δ)

            _d2u[i,j,k] = Δ
        end
    end

    return d2u
end

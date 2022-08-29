# TODO: input and document
__DEFAULT_LBV_PRE(nlevels::Integer) = ntuple(nlevels-1) do l
    Smoothers(
        RedBlackGaussSeidel(ForwardSweep(), iter=2^(l-1)),
        RedBlackGaussSeidel(BoundarySweep(ForwardSweep()), iter=2^l),
    )
end

__DEFAULT_LBV_COARSE() = Smoothers(
    RedBlackGaussSeidel(ForwardSweep(), iter=512),
    RedBlackGaussSeidel(BackwardSweep(), iter=512),
)

__DEFAULT_LBV_POST(nlevels::Integer) = ntuple(nlevels-1) do l
    Smoothers(
        RedBlackGaussSeidel(BoundarySweep(BackwardSweep()), iter=2^l),
        RedBlackGaussSeidel(BackwardSweep(), iter=2^(l-1)),
    )
end

"""
    lbv(
        f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        atol::Real = sqrt(eps(Float64)),
        rtol::Real = sqrt(eps(Float64)),
        maxit::Integer = maximum(size(f)),
        verbose::Bool = false
    ) -> typeof(similar(f))

Laplacian boundary value problem (LBV) [1].

The Laplacian is computed using second order central finite differences.
The resulting Poisson's equation is then solved inside an ROI (`mask`) with
homogenous Dirichlet boundary condition (BC) using a multigrid-preconditioned
conjugate gradient method. The boundary of the ROI is set such that values
outside of it (`mask = 0`) are taken as boundary points and values inside of it
(`mask = 1`) as interior points, ie. BC: `fl[!mask] = 0`.

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size

### Keywords
- `atol::Real = sqrt(eps(Float64))`: absolute stopping tolerance
- `rtol::Real = sqrt(eps(Float64))`: relative stopping tolerance
- `maxit::Integer = maximum(size(f))`: maximum number of cg iterations
- `verbose::Bool = false`: print convergence information

### Returns
- `typeof(similar(f))`: background corrected local field/phase

### References
[1] Zhou D, Liu T, Spincemaille P, Wang Y. Background field removal by solving
    the Laplacian boundary value problem. NMR in Biomedicine. 2014 Mar;27(3):312-9.
"""
function lbv(
    f::AbstractArray{<:AbstractFloat, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    atol::Real = sqrt(eps(Float64)),
    rtol::Real = sqrt(eps(Float64)),
    maxit::Integer = maximum(size(f)),
    verbose::Bool = false
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _lbv!(tzero(f), f, mask, vsz, atol, rtol, maxit, verbose)
end

function _lbv!(
    fl::AbstractArray{<:AbstractFloat, N},
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    atol::Real,
    rtol::Real,
    maxit::Integer,
    verbose::Bool,
) where {T<:AbstractFloat, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    checkshape(fl, f, (:fl, :f))
    checkshape(axes(mask), axes(f)[1:3], (:mask, :f))

    # crop to avoid unnecessary work. fl[!mask] = 0
    sz = size(mask)
    Rc = crop_indices(mask)

    # pad for finite diffy stencil
    I1, I2 = first(Rc), last(Rc)
    Rc = CartesianIndices(ntuple(ndims(Rc)) do n
        max(1, I1[n]-1) : min(sz[n], I2[n]+1)
    end)

    # TODO: performance test proper cutoff
    szc = size(Rc)
    _crop = maximum(sz .- szc) > 10

    if !_crop
        mc = mask
        fc = f
        flc = fl
    else
        mc = similar(mask, szc)
        mc = tcopyto!(mc, @view(mask[Rc]))

        if N == 3
            fc = @view(f[Rc])
            flc = similar(fl, szc)
            flc = tcopyto!(flc, @view(fl[Rc]))
        else
            fc = @view(f[Rc,:])
            flc = similar(fl, (szc..., size(fl, 4)))
            flc = tcopyto!(flc, @view(fl[Rc,:]))
        end
    end

    # set boundaries
    for t in axes(f, 4)
        fct = @view(f[Rc,t])
        @bfor fct[I] *= mc[I]
    end

    # Laplacian
    A = Poisson2(T, mc, vsz)
    d2fc = A*fc

    # MGPCG options
    nlevels = __DEFAULT_MGPCG_DEPTH(mc) # stc/utils/poisson_solver/mgpcg.jl

    opts = (
        presmoother  = __DEFAULT_LBV_PRE(nlevels),
        coarsesolver = __DEFAULT_LBV_COARSE(),
        postsmoother = __DEFAULT_LBV_POST(nlevels),
        atol = atol,
        rtol = rtol,
        maxit = maxit,
        verbose = verbose,
    )

    # solve
    flc = mgpcg!(flc, A, d2fc; opts...)

    if _crop
        if N == 3
            tcopyto!(@view(fl[Rc]), flc)
        else
            tcopyto!(@view(fl[Rc,:]), flc)
        end
    end

    return fl
end

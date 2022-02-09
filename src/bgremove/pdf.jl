"""
    pdf(
        f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        W::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, N)}} = nothing,
        pad::NTuple{3, Integer} = (0, 0, 0),
        bdir::NTuple{3, Real} = (0, 0, 1),
        Dkernel::Symbol = :i,
        lambda::Real = 0,
        tol::Real = 1e-5,
        maxit::Integer = ceil(sqrt(numel(mask))),
        verbose::Bool = false
    ) typeof(similar(f))

Projection onto dipole fields (PDF) [1].

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size for dipole kernel

### Keywords
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, N)}} = nothing`:
    data fidelity weights
- `pad::NTuple{3, Integer} = (0, 0, 0)`: zero padding array
    - `< 0`: no padding
    - `≥ 0`: minimum padding to fast fft size
- `bdir::NTuple{3, Real} = (0, 0, 1)`: unit vector of B field direction
- `Dkernel::Symbol = :i`: dipole kernel method
- `lambda::Real = 0`: regularization parameter
- `tol::Real = 1e-5`: stopping tolerance for iterative solver
- `maxit::Integer = ceil(sqrt(length(mask)))`: maximum number of iterations for
    iterative solver
- `verbose::Bool = false`: print convergence information

### Returns
- `typeof(similar(f))`: background corrected local field/phase

### References
[1] Liu T, Khalidov I, de Rochefort L, Spincemaille P, Liu J, Tsiouris AJ,
    Wang Y. A novel background field removal method for MRI using projection
    onto dipole fields. NMR in Biomedicine. 2011 Nov;24(9):1129-36.
"""
function pdf(
    f::AbstractArray{<:AbstractFloat, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    W::Union{Nothing, AbstractArray{<:AbstractFloat}} = nothing,
    pad::NTuple{3, Integer} = (0, 0, 0),
    bdir::NTuple{3, Real} = (0, 0, 1),
    Dkernel::Symbol = :i,
    lambda::Real = 0,
    tol::Real = 1e-5,
    maxit::Integer = ceil(Int, sqrt(length(mask))),
    verbose::Bool = false
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _pdf!(
        tzero(f), f, mask, vsz, W, pad, bdir, Dkernel, lambda, tol, maxit, verbose
    )
end

function _pdf!(
    fl::AbstractArray{<:AbstractFloat, N},
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    W::Union{Nothing, AbstractArray{<:AbstractFloat}},
    pad::NTuple{3, Integer},
    bdir::NTuple{3, Real},
    Dkernel::Symbol,
    lambda::Real,
    tol::Real,
    maxit::Integer,
    verbose::Bool = false
) where {T<:AbstractFloat, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    size(fl) == size(f) || throw(DimensionMismatch())
    size(mask) == size(f)[1:3] || throw(DimensionMismatch())

    if W !== nothing
        ndims(W) ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(M)d"))
        size(W)[1:3] == size(f)[1:3] || throw(DimensionMismatch())
        ndims(W) == 4 && size(W, 4) == size(f, 4) || throw(DimensionMismatch())
    end

    Dkernel ∈ (:k, :kspace, :i, :ispace) ||
        throw(ArgumentError("Dkernel must be one of :k, :kspace, :i, :ispace, got :$(Dkernel)"))

    # pad to fast fft size
    fp = padfastfft(@view(f[:,:,:,1]), pad, rfft=true)
    m = padfastfft(mask, pad, rfft=true)

    # init vars and fft plans
    sz0 = size(mask)
    sz = size(m)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    M̃ = similar(m)
    MW = similar(fp)

    D = Array{T, 3}(undef, sz_)
    F̂ = Array{complex(T), 3}(undef, sz_)

    FFTW.set_num_threads(num_cores())
    P = plan_rfft(fp)
    iP = inv(P)

    # LSMR setup
    # move up here so we can use LSMR's `b` as a tmp variable
    A = LinearMap{T}(
        (Av, v) -> _A_pdf!(Av, v, MW, iP, D, F̂, P, M̃),
        (Atu, u) -> _At_pdf!(Atu, u, M̃, iP, D, F̂, P, MW),
        length(fp),
        issymmetric = false,
        ishermitian = false,
        isposdef = true,
        ismutating = true
    )

    WS = LSMRWorkspace(A)
    b  = reshape(WS.b, sz)

    # get dipole kernel
    D = _dipole_kernel!(D, F̂, b, sz0, vsz, bdir, P, Dkernel, :rfft)

    # background mask
    @inbounds @batch for I in eachindex(m)
        M̃[I] = !m[I]
    end

    # pre-compute mask*weights
    if W === nothing
        # no weights
        MW = _tcopyto!(MW, m)

    elseif ndims(W) == 3
        # same weights for all echoes
        b = padarray!(b, W)
        @inbounds @batch for I in eachindex(MW)
            MW[I] = m[I] * b[I]
        end

    elseif ndims(W) == 4
        # weights for each echo. compute inside for loop
        # VOID
    end

    @inbounds for t in axes(f, 4)
        if verbose && size(f, 4) > 1
            @printf("Echo: %d/%d\n", t, size(f, 4))
        end

        if t > 1
            fp = padarray!(fp, @view(f[:,:,:,t]))
        end

        # mask * weight if we can't precompute
        if W !== nothing
            if ndims(W) == 4
                b = padarray!(b, @view(W[:,:,:,t]))
                @batch for I in eachindex(MW)
                    MW[I] = m[I] * b[I]
                end
            end
        end

        # rhs
        @batch for I in eachindex(b)
            b[I] = MW[I] * fp[I]
        end

        # solve
        lsmr!(WS; lambda=lambda, atol=tol, btol=tol, maxit=maxit, verbose=verbose)

        # compute background fields
        xb = reshape(WS.x, sz)
        @batch for I in eachindex(xb)
            xb[I] *= M̃[I]
        end

        F̂ = mul!(F̂, P, xb)
        @batch for I in eachindex(F̂)
            F̂[I] *= D[I]
        end

        xb = mul!(xb, iP, F̂)
        @batch for I in eachindex(fp)
            fp[I] = m[I] * (fp[I] - xb[I])
        end

        if verbose
            println()
        end

        unpadarray!(@view(fl[:,:,:,t]), fp)
    end

    return fl
end


function _A_pdf!(Av, v, W, iP, D, F̂, P, M̃)
    v = reshape(v, size(W))
    x = reshape(Av, size(W))

    @inbounds @batch for I in eachindex(x)
        x[I] = M̃[I] * v[I]
    end

    F̂ = mul!(F̂, P, x)
    @inbounds @batch for I in eachindex(F̂)
        F̂[I] *= D[I]
    end

    x = mul!(x, iP, F̂)
    @inbounds @batch for I in eachindex(x)
        x[I] *= W[I]
    end

    return Av
end


function _At_pdf!(Atu, u, M̃, iP, D, F̂, P, W)
    u = reshape(u, size(W))
    x = reshape(Atu, size(W))

    @inbounds @batch for I in eachindex(x)
        x[I] = W[I] * u[I]
    end

    F̂ = mul!(F̂, P, x)
    @inbounds @batch for I in eachindex(F̂)
        F̂[I] *= D[I] # conj(D), D is real
    end

    x = mul!(x, iP, F̂)
    @inbounds @batch for I in eachindex(x)
        x[I] *= M̃[I]
    end

    return Atu
end

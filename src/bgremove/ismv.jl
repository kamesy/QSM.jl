"""
    function ismv(
        f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        r::Real = 2*maximum(vsz),
        tol::Real = 1e-3,
        maxit::Integer = 500,
        verbose::Bool = false,
    ) -> Tuple{typeof(similar(f)), typeof(similar(mask))}

Iterative spherical mean value method (iSMV) [1].

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size for smv kernel

### Keywords
- `r::Real = 2*maximum(vsz)`: radius of smv kernel in units of `vsz`
- `tol::Real = 1e-3`: stopping tolerance
- `maxit::Integer = 500`: maximum number of iterations
- `verbose::Bool = false`: print convergence information

### Returns
- `typeof(similar(f))`: background corrected local field/phase
- `typeof(similar(mask))`: eroded binary mask

### References
[1] Wen Y, Zhou D, Liu T, Spincemaille P, Wang Y. An iterative spherical mean
    value method for background field removal in MRI.  Magnetic resonance in
    medicine. 2014 Oct;72(4):1065-71.
"""
function ismv(
    f::AbstractArray{<:AbstractFloat, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    r::Real = 2*maximum(vsz),
    tol::Real = 1e-3,
    maxit::Integer = 500,
    verbose::Bool = false,
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _ismv!(tzero(f), tzero(mask), f, mask, vsz, r, tol, maxit, verbose)
end

function _ismv!(
    fl::AbstractArray{<:AbstractFloat, N},
    smask::AbstractArray{Bool, 3},
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    r::Real,
    tol::Real,
    maxit::Integer,
    verbose::Bool,
) where {T<:AbstractFloat, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    checkshape(fl, f, (:fl, :f))
    checkshape(smask, mask, (:smask, :mask))
    checkshape(axes(mask), axes(f)[1:3], (:mask, :f))

    # crop image and pad for convolution
    Rc = crop_indices(mask)
    pad = ntuple(_ -> Int(cld(r, minimum(vsz))), Val(3))

    fp = padfastfft(@view(f[Rc,1]), pad, rfft=true)
    m = padfastfft(@view(mask[Rc]), pad, rfft=true)

    # init vars and fft plans
    sz = size(m)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    s = similar(fp)
    fb = similar(fp)
    bc = similar(fp)
    m0 = similar(m)

    S = Array{T}(undef, sz_)
    F̂ = Array{complex(T)}(undef, sz_)

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(fp)
    iP = inv(P)

    # get smv kernel
    S = _smv_kernel!(S, F̂, s, vsz, r, P)

    # constants
    δ = one(eltype(s)) - sqrt(eps(eltype(s)))

    # erode mask
    s = tcopyto!(s, m) # in-place type conversion, reuse smv var
    m0 = tcopyto!(m0, m)

    F̂ = mul!(F̂, P, s)
    @bfor F̂[I] *= S[I]

    s = mul!(s, iP, F̂)
    @bfor m[I] = s[I] > δ

    # iSMV
    for t in axes(f, 4)
        if verbose && size(f, 4) > 1
            @printf("Echo: %d/%d\n", t, size(f, 4))
        end

        if t > 1
            fp = padarray!(fp, @view(f[Rc,t]))
        end

        @bfor begin
            bc[I] = (m0[I] - m[I]) * fp[I]
            fb[I] = fp[I]
        end

        fb = __ismv!(fb, s, S, bc, m, iP, F̂, P, tol, maxit, verbose)

        @bfor fp[I] = m[I] * (fp[I] - fb[I])

        unpadarray!(@view(fl[Rc,t]), fp)
    end

    unpadarray!(@view(smask[Rc]), m)

    return fl, smask
end


function __ismv!(f::AbstractArray{T}, f0, S, bc, m, iP, F̂, P, tol, maxit, verbose) where {T}
    @bfor f0[I] = m[I] * f[I]

    nr = norm(f0)
    ϵ = tol * nr

    if verbose
        @printf("iter%6s\tresnorm\n", "")
    end

    for i in 1:maxit
        if nr ≤ ϵ
            break
        end

        F̂ = mul!(F̂, P, f)
        @bfor F̂[I] *= S[I]

        f = mul!(f, iP, F̂)
        @batch threadlocal=zero(T)::T for I in eachindex(f)
            f[I] = muladd(m[I], f[I], bc[I])

            # norm(f0 - f)
            f0_f = f0[I] - f[I]
            threadlocal = muladd(f0_f, f0_f, threadlocal)
            f0[I] = f[I]
        end

        nr = sqrt(sum(threadlocal::Vector{T}))

        if verbose
            @printf("%5d/%d\t%1.3e\n", i, maxit, nr)
        end
    end

    if verbose
        println()
    end

    return f
end

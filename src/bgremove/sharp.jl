#####
##### SHARP
#####

"""
    sharp(
        f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        r::Real = 18*minimum(vsz),
        thr::Real = 0.05,
    ) -> Tuple{typeof(similar(f)), typeof(similar(mask))}

Sophisticated harmonic artifact reduction for phase data (SHARP) [1].

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size for smv kernel

### Keywords
- `r::Real = 18*minimum(vsz)`: radius of smv kernel in units of `vsz`
- `thr::Real = 0.05`: threshold for high pass filter

### Returns
- `typeof(similar(f))`: background corrected local field/phase
- `typeof(similar(mask))`: eroded binary mask

### References
[1] Schweser F, Deistung A, Lehr BW, Reichenbach JR. Quantitative imaging of
    intrinsic magnetic tissue properties using MRI signal phase: an approach to
    in vivo brain iron metabolism?. Neuroimage. 2011 Feb 14;54(4):2789-807.
"""
function sharp(
    f::AbstractArray{<:AbstractFloat, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    r::Real = 18*minimum(vsz),
    thr::Real = 0.05,
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _sharp!(tzero(f), tzero(mask), f, mask, vsz, r, thr)
end

function _sharp!(
    fl::AbstractArray{<:AbstractFloat, N},
    smask::AbstractArray{Bool, 3},
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    r::Real,
    thr::Real
) where {T<:AbstractFloat, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    checkshape(fl, f, (:fl, :f))
    checkshape(smask, mask, (:smask, :mask))
    checkshape(axes(mask), axes(f)[1:3], (:mask, :f))

    # crop image and pad for convolution
    Rc = crop_indices(mask)
    pad = ntuple(_ -> Int(cld(r, minimum(vsz))), Val(3))

    fp = padfastfft(@view(f[Rc,1]), pad, rfft=true)
    m  = padfastfft(@view(mask[Rc]), pad, rfft=true)

    # init vars and fft plans
    sz = size(m)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    s = Array{T}(undef, sz)
    S = Array{T}(undef, sz_)
    F̂ = Array{complex(T)}(undef, sz_)

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(fp)
    iP = inv(P)

    # get smv kernel
    S = _smv_kernel!(S, F̂, s, vsz, r, P)

    # constants
    _thr  = convert(eltype(S), thr)
    _one  = one(eltype(S))
    _zero = zero(eltype(F̂))
    _δ    = one(eltype(s)) - sqrt(eps(eltype(s)))

    # erode mask
    s = tcopyto!(s, m) # in-place type conversion, reuse smv var

    F̂ = mul!(F̂, P, s)
    @inbounds @batch for I in eachindex(F̂)
        F̂[I] *= S[I]
        S[I] = _one - S[I]
    end

    s = mul!(s, iP, F̂)
    @inbounds @batch for I in eachindex(m)
        m[I] = s[I] > _δ
    end

    # SHARP
    @inbounds for t in axes(f, 4)
        if t > 1
            fp = padarray!(fp, @view(f[Rc,t]))
        end

        F̂ = mul!(F̂, P, fp)
        @batch for I in eachindex(F̂)
            F̂[I] *= S[I]
        end

        fp = mul!(fp, iP, F̂)
        @batch for I in eachindex(fp)
            fp[I] *= m[I]
        end

        F̂ = mul!(F̂, P, fp)
        @batch for I in eachindex(F̂)
            F̂[I] = ifelse(abs(S[I]) < _thr, _zero, F̂[I]*inv(S[I]))
        end

        fp = mul!(fp, iP, F̂)
        @batch for I in eachindex(fp)
            fp[I] *= m[I]
        end

        unpadarray!(@view(fl[Rc,t]), fp)
    end

    unpadarray!(@view(smask[Rc]), m)

    return fl, smask
end


#####
##### V-SHARP
#####

"""
    vsharp(
        f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        r::AbstractVector{<:Real} = 18*minimum(vsz):-2*maximum(vsz):2*maximum(vsz),
        thr::Real = 0.05,
    ) -> Tuple{typeof(similar(f)), typeof(similar(mask))}

Variable kernels sophisticated harmonic artifact reduction for phase data (V-SHARP) [1].

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size for smv kernel

### Keywords
- `r::AbstractVector{<:Real} = 18*minimum(vsz):-2*maximum(vsz):2*maximum(vsz)`:
    radii of smv kernels in mm
- `thr::Real = 0.05`: threshold for high pass filter

### Returns
- `typeof(similar(f))`: background corrected local field/phase
- `typeof(similar(mask))`: eroded binary mask

### References
[1] Wu B, Li W, Guidon A, Liu C. Whole brain susceptibility mapping using
    compressed sensing. Magnetic resonance in medicine. 2012 Jan;67(1):137-47.
"""
function vsharp(
    f::AbstractArray{<:AbstractFloat, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    r::AbstractVector{<:Real} = 18*minimum(vsz):-2*maximum(vsz):2*maximum(vsz),
    thr::Real = 0.05,
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return length(r) == 1 ?
        _sharp!(tzero(f), tzero(mask), f, mask, vsz, first(r), thr) :
        _sharp!(tzero(f), tzero(mask), f, mask, vsz, r, thr)
end

function _sharp!(
    fl::AbstractArray{<:AbstractFloat, 3},
    smask::AbstractArray{Bool, 3},
    f::AbstractArray{T, 3},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    r::AbstractVector{<:Real},
    thr::Real,
) where {T<:AbstractFloat}
    checkshape(fl, f, (:fl, :f))
    checkshape(smask, mask, (:smask, :mask))
    checkshape(mask, f, (:mask, :f))
    !isempty(r) || throw(ArgumentError("r must not be empty"))

    rs = sort!(unique(collect(r)), rev=true)

    # crop image and pad for convolution
    Rc = crop_indices(mask)
    pad = ntuple(_ -> Int(cld(maximum(rs), minimum(vsz))), Val(3))

    fp = padfastfft(@view(f[Rc]), pad, rfft=true)
    mr = padfastfft(@view(mask[Rc]), pad, rfft=true)

    # init vars and fft plans
    sz = size(mr)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    s = Array{T}(undef, sz)
    m = Array{Bool}(undef, sz)

    m = tfill!(m, 0)

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(s)
    iP = inv(P)

    S  = Array{T}(undef, sz_)
    iS = Array{T}(undef, sz_)
    F̂  = Array{complex(T)}(undef, sz_)
    M̂  = Array{complex(T)}(undef, sz_)
    F̂p = Array{complex(T)}(undef, sz_)

    F̂p = mul!(F̂p, P, fp)
    fp = tfill!(fp, 0)

    # constants
    _thr  = convert(eltype(S), thr)
    _one  = one(eltype(S))
    _zero = zero(eltype(F̂))
    _δ    = one(eltype(s)) - sqrt(eps(eltype(s)))

    # fft of original mask
    s = tcopyto!(s, mr) # in-place type conversion
    M̂ = mul!(M̂, P, s)

    @inbounds for (i, r) in enumerate(rs)
        # get smv kernel
        S = _smv_kernel!(S, F̂, s, vsz, r, P)

        # erode mask
        @batch for I in eachindex(F̂)
            F̂[I] = S[I] * M̂[I]
            S[I] = _one - S[I]
        end

        s = mul!(s, iP, F̂)
        @batch for I in eachindex(mr)
            mr[I] = s[I] > _δ
        end

        # high-pass filter first (largest) kernel for deconvolution
        if i == 1
            @batch for I in eachindex(iS)
                iS[I] = ifelse(abs(S[I]) < _thr, _zero, inv(S[I]))
            end
        end

        # SHARP
        @batch for I in eachindex(F̂)
            F̂[I] = S[I] * F̂p[I]
        end

        s = mul!(s, iP, F̂)
        @batch for I in eachindex(fp)
            if mr[I] && !m[I]
                fp[I] = s[I]
            end
        end

        m, mr = mr, m
    end

    # deconvolution + high-pass filter
    F̂p = mul!(F̂p, P, fp)
    @batch for I in eachindex(F̂p)
        F̂p[I] *= iS[I]
    end

    fp = mul!(fp, iP, F̂p)
    @batch for I in eachindex(fp)
        fp[I] *= m[I]
    end

    unpadarray!(@view(fl[Rc]), fp)
    unpadarray!(@view(smask[Rc]), m)

    return fl, smask
end

function _sharp!(
    fl::AbstractArray{<:AbstractFloat, 4},
    smask::AbstractArray{Bool, 3},
    f::AbstractArray{T, 4},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    r::AbstractVector{<:Real},
    thr::Real,
) where {T<:AbstractFloat}
    checkshape(fl, f, (:fl, :f))
    checkshape(smask, mask, (:smask, :mask))
    checkshape(axes(mask), axes(f)[1:3], (:mask, :f))
    length(r) > 0 || throw(ArgumentError("r must not be empty"))

    rs = sort!(unique(collect(r)), rev=true)

    # crop image and pad for convolution
    Rc = crop_indices(mask)
    pad = ntuple(_ -> Int(cld(maximum(rs), minimum(vsz))), Val(3))

    fp = padfastfft(@view(f[Rc,:]), pad, rfft=true)
    mr = padfastfft(@view(mask[Rc]), pad, rfft=true)

    # init vars and fft plans
    sz = size(mr)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    s = Array{T}(undef, sz)
    m = Array{Bool}(undef, sz)

    m = tfill!(m, 0)

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(s)
    iP = inv(P)

    P4 = plan_rfft(fp, 1:3)
    iP4 = inv(P4)

    S  = Array{T}(undef, sz_)
    iS = Array{T}(undef, sz_)
    F̂  = Array{complex(T)}(undef, sz_)
    M̂  = Array{complex(T)}(undef, sz_)
    F̂p = Array{complex(T)}(undef, (sz_..., size(fp, 4)))

    F̂p = mul!(F̂p, P4, fp)
    fp = tfill!(fp, 0)

    # constants
    _thr  = convert(eltype(S), thr)
    _one  = one(eltype(S))
    _zero = zero(eltype(F̂))
    _δ    = one(eltype(s)) - sqrt(eps(eltype(s)))

    # fft of original mask
    s = tcopyto!(s, mr) # in-place type conversion
    M̂ = mul!(M̂, P, s)

    @inbounds for (i, r) in enumerate(rs)
        # get smv kernel
        S = _smv_kernel!(S, F̂, s, vsz, r, P)

        # erode mask
        @batch for I in eachindex(F̂)
            F̂[I] = S[I] * M̂[I]
            S[I] = _one - S[I]
        end

        s = mul!(s, iP, F̂)
        @batch for I in eachindex(mr)
            mr[I] = s[I] > _δ
        end

        # high-pass filter first (largest) kernel for deconvolution
        if i == 1
            @batch for I in eachindex(iS)
                iS[I] = ifelse(abs(S[I]) < _thr, _zero, inv(S[I]))
            end
        end

        # SHARP
        for t in axes(fp, 4)
            F̂t = @view(F̂p[:,:,:,t])
            ft = @view(fp[:,:,:,t])

            @batch for I in eachindex(F̂)
                F̂[I] = S[I] * F̂t[I]
            end

            s = mul!(s, iP, F̂)
            @batch for I in eachindex(ft)
                if mr[I] && !m[I]
                    ft[I] = s[I]
                end
            end
        end

        m, mr = mr, m
    end

    # deconvolution + high-pass filter
    F̂p = mul!(F̂p, P4, fp)
    @inbounds for t in axes(F̂p, 4)
        F̂t = @view(F̂p[:,:,:,t])
        @batch for I in eachindex(F̂t)
            F̂t[I] *= iS[I]
        end
    end

    fp = mul!(fp, iP4, F̂p)
    @inbounds for t in axes(fp, 4)
        ft = @view(fp[:,:,:,t])
        @batch for I in eachindex(ft)
            ft[I] *= m[I]
        end
    end

    unpadarray!(@view(fl[Rc,:]), fp)
    unpadarray!(@view(smask[Rc]), m)

    return fl, smask
end

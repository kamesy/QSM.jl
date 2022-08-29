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
    thrT  = convert(eltype(S), thr)
    oneT  = one(eltype(S))
    zeroT = zero(eltype(F̂))
    δ     = one(eltype(s)) - sqrt(eps(eltype(s)))

    # erode mask
    s = tcopyto!(s, m) # in-place type conversion, reuse smv var

    F̂ = mul!(F̂, P, s)
    @bfor begin
        F̂[I] *= S[I]
        S[I]  = oneT - S[I]
    end

    s = mul!(s, iP, F̂)
    @bfor m[I] = s[I] > δ

    # SHARP
    for t in axes(f, 4)
        if t > 1
            fp = padarray!(fp, @view(f[Rc,t]))
        end

        F̂ = mul!(F̂, P, fp)
        @bfor F̂[I] *= S[I]

        fp = mul!(fp, iP, F̂)
        @bfor fp[I] *= m[I]

        F̂ = mul!(F̂, P, fp)
        @bfor F̂[I] = ifelse(abs(S[I]) < thrT, zeroT, F̂[I]*inv(S[I]))

        fp = mul!(fp, iP, F̂)
        @bfor fp[I] *= m[I]

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
    thrT  = convert(eltype(S), thr)
    oneT  = one(eltype(S))
    zeroT = zero(eltype(F̂))
    δ     = one(eltype(s)) - sqrt(eps(eltype(s)))

    # fft of original mask
    s = tcopyto!(s, mr) # in-place type conversion
    M̂ = mul!(M̂, P, s)

    for (i, r) in enumerate(rs)
        # get smv kernel
        S = _smv_kernel!(S, F̂, s, vsz, r, P)

        # erode mask
        @bfor begin
            F̂[I] = S[I] * M̂[I]
            S[I] = oneT - S[I]
        end

        s = mul!(s, iP, F̂)
        @bfor mr[I] = s[I] > δ

        # high-pass filter first (largest) kernel for deconvolution
        if i == 1
            @bfor iS[I] = ifelse(abs(S[I]) < thrT, zeroT, inv(S[I]))
        end

        # SHARP
        @bfor F̂[I] = S[I] * F̂p[I]

        s = mul!(s, iP, F̂)
        @bfor if mr[I] && !m[I]
            fp[I] = s[I]
        end

        m, mr = mr, m
    end

    # deconvolution + high-pass filter
    F̂p = mul!(F̂p, P, fp)
    @bfor F̂p[I] *= iS[I]

    fp = mul!(fp, iP, F̂p)
    @bfor fp[I] *= m[I]

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
    !isempty(r) || throw(ArgumentError("r must not be empty"))

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
    thrT  = convert(eltype(S), thr)
    oneT  = one(eltype(S))
    zeroT = zero(eltype(F̂))
    δ     = one(eltype(s)) - sqrt(eps(eltype(s)))

    # fft of original mask
    s = tcopyto!(s, mr) # in-place type conversion
    M̂ = mul!(M̂, P, s)

    for (i, r) in enumerate(rs)
        # get smv kernel
        S = _smv_kernel!(S, F̂, s, vsz, r, P)

        # erode mask
        @bfor begin
            F̂[I] = S[I] * M̂[I]
            S[I] = oneT - S[I]
        end

        s = mul!(s, iP, F̂)
        @bfor mr[I] = s[I] > δ

        # high-pass filter first (largest) kernel for deconvolution
        if i == 1
            @bfor iS[I] = ifelse(abs(S[I]) < thrT, zeroT, inv(S[I]))
        end

        # SHARP
        for t in axes(fp, 4)
            F̂t = @view(F̂p[:,:,:,t])
            ft = @view(fp[:,:,:,t])

            @bfor F̂[I] = S[I] * F̂t[I]

            s = mul!(s, iP, F̂)
            @bfor if mr[I] && !m[I]
                ft[I] = s[I]
            end
        end

        m, mr = mr, m
    end

    # deconvolution + high-pass filter
    F̂p = mul!(F̂p, P4, fp)
    for t in axes(F̂p, 4)
        F̂t = @view(F̂p[:,:,:,t])
        @bfor F̂t[I] *= iS[I]
    end

    fp = mul!(fp, iP4, F̂p)
    for t in axes(fp, 4)
        ft = @view(fp[:,:,:,t])
        @bfor ft[I] *= m[I]
    end

    unpadarray!(@view(fl[Rc,:]), fp)
    unpadarray!(@view(smask[Rc]), m)

    return fl, smask
end

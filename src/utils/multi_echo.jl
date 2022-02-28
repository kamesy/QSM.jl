#####
##### Linear
#####

"""
    fit_echo_linear(
        phas::AbstractArray{<:AbstractFloat, 4},
        W::AbstractArray{<:AbstractFloat, 4},
        TEs::NTuple{N, Real};
        phase_offset::Bool = true
    ) -> Tuple{typeof(similar(phas)){3}, [typeof(similar(phas)){3}]}

Weighted least squares for multi-echo data.

### Arguments
- `phas::AbstractArray{<:AbstractFloat, 4}`: unwrapped multi-echo phase
- `W::AbstractArray{<:AbstractFloat, 4}`: reciprocal of error variance of voxel
- `TEs::NTuple{N, Real}`: echo times

### Keywords
- `phase_offset::Bool = true`: model phase offset (`true`)

### Returns
- `typeof(similar(phas)){3}`: weighted least-squares estimate for phase
- [`typeof(similar(phas)){3}`]: weighted least-squares estimate for phase offset
    if `phase_offset = true`
"""
function fit_echo_linear(
    phas::AbstractArray{<:AbstractFloat, 4},
    W::AbstractArray{<:AbstractFloat, 4},
    TEs::NTuple{N, Real};
    phase_offset::Bool = true
) where {N}
    p = tzero(phas, size(phas)[1:3])
    if !phase_offset
        return fit_echo_linear!(p, phas, W, TEs)
    else
        return fit_echo_linear!(p, tzero(p), phas, W, TEs)
    end
end

"""
    fit_echo_linear!(
        p::AbstractArray{<:AbstractFloat, 3},
        phas::AbstractArray{<:AbstractFloat, 4},
        W::AbstractArray{<:AbstractFloat, 4},
        TEs::NTuple{N, Real}
    ) -> p

Weighted least squares for multi-echo data (phase offset = 0).

### Arguments
- `p::AbstractArray{<:AbstractFloat, 3}`: weighted least-squares estimate for phase
- `phas::AbstractArray{<:AbstractFloat, 4}`: unwrapped multi-echo phase
- `W::AbstractArray{<:AbstractFloat, 4}`: reciprocal of error variance of voxel
- `TEs::NTuple{N, Real}`: echo times

### Returns
- `p`: weighted least-squares estimate for phase
"""
function fit_echo_linear!(
    p::AbstractArray{Tp, 3},
    phas::AbstractArray{Tphas, 4},
    W::AbstractArray{TW, 4},
    TEs::NTuple{NT, Real}
) where {Tp<:AbstractFloat, Tphas<:AbstractFloat, TW<:Real, NT}
    nx, ny, nz, nt = size(phas)

    nt == NT || throw(DimensionMismatch())
    size(p) == (nx, ny, nz) || throw(DimensionMismatch())
    size(W) == size(phas)   || throw(DimensionMismatch())

    T = promote_type(Tp, Tphas, TW)
    tes = convert.(T, TEs)

    _zeroT = zero(T)
    _zeroTp = zero(Tp)

    @threads for k in 1:nz
        @inbounds for j in 1:ny
            for i in 1:nx
                num = _zeroT
                den = _zeroT
                for t in Base.OneTo(NT)
                    w = W[i,j,k,t] * W[i,j,k,t] * tes[t]
                    num = muladd(w, phas[i,j,k,t], num)
                    den = muladd(w, tes[t], den)
                end
                p[i,j,k] = iszero(den) ? _zeroTp : num * inv(den)
            end
        end
    end

    return p
end

"""
    fit_echo_linear!(
        p::AbstractArray{<:AbstractFloat, 3},
        p0::AbstractArray{<:AbstractFloat, 3},
        phas::AbstractArray{<:AbstractFloat, 4},
        W::AbstractArray{<:AbstractFloat, 4},
        TEs::NTuple{N, Real}
    ) -> (p, p0)

Weighted least squares for multi-echo data (estimate phase offset).

### Arguments
- `p::AbstractArray{<:AbstractFloat, 3}`: weighted least-squares estimate for phase
- `p0::AbstractArray{<:AbstractFloat, 3}`: weighted least-squares estimate for phase offset
- `phas::AbstractArray{<:AbstractFloat, 4}`: unwrapped multi-echo phase
- `W::AbstractArray{<:AbstractFloat, 4}`: reciprocal of error variance of voxel
- `TEs::NTuple{N, Real}`: echo times

### Returns
- `p`: weighted least-squares estimate for phase
- `p0`: weighted least-squares estimate for phase offset
"""
function fit_echo_linear!(
    p::AbstractArray{Tp, 3},
    p0::AbstractArray{Tp0, 3},
    phas::AbstractArray{Tphas, 4},
    W::AbstractArray{TW, 4},
    TEs::NTuple{NT, Real}
) where {Tp<:AbstractFloat, Tp0<:AbstractFloat, Tphas<:AbstractFloat, TW<:Real, NT}
    nx, ny, nz, nt = size(phas)

    nt == NT || throw(DimensionMismatch())
    size(p)  == (nx, ny, nz) || throw(DimensionMismatch())
    size(p0) == (nx, ny, nz) || throw(DimensionMismatch())
    size(W)  == size(phas)   || throw(DimensionMismatch())

    T = promote_type(Tp, Tp0, Tphas, TW)
    tes = convert.(T, TEs)

    _zeroT = zero(T)
    _zeroTp = zero(Tp)

    @threads for k in 1:nz
        @inbounds for j in 1:ny
            for i in 1:nx
                x = _zeroT
                y = _zeroT
                w = _zeroT
                for t in Base.OneTo(NT)
                    w += W[i,j,k,t]
                    x = muladd(W[i,j,k,t], tes[t], x)
                    y = muladd(W[i,j,k,t], phas[i,j,k,t], y)
                end

                w = inv(w)
                x *= w
                y *= w

                num = _zeroT
                den = _zeroT
                for t in Base.OneTo(NT)
                    xx = tes[t] - x
                    yy = phas[i,j,k,t] - y
                    ww = W[i,j,k,t] * xx
                    num = muladd(ww, yy, num)
                    den = muladd(ww, xx, den)
                end

                p[i,j,k] = iszero(den) ? _zeroTp : num * inv(den)
                p0[i,j,k] = y - p[i,j,k] * x
            end
        end
    end

    return p, p0
end

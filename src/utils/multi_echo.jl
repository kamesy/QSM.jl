#####
##### Linear
#####

"""
    fit_echo_linear(
        phas::AbstractArray{<:AbstractFloat, N > 1},
        W::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real};
        phase_offset::Bool = true
    ) -> Tuple{typeof(similar(phas)){N-1}, [typeof(similar(phas)){N-1}]}

Weighted least squares for multi-echo data.

### Arguments
- `phas::AbstractArray{<:AbstractFloat, N > 1}`: unwrapped multi-echo phase
- `W::AbstractArray{<:AbstractFloat, N > 1}`: reciprocal of error variance of voxel
- `TEs::AbstractVector{<:Real}`: echo times

### Keywords
- `phase_offset::Bool = true`: model phase offset (`true`)

### Returns
- `typeof(similar(phas)){N-1}`: weighted least-squares estimate for phase
- [`typeof(similar(phas)){N-1}`]: weighted least-squares estimate for phase offset
    if `phase_offset = true`
"""
function fit_echo_linear(
    phas::AbstractArray{<:AbstractFloat, N},
    W::AbstractArray{<:AbstractFloat, N},
    TEs::AbstractVector{<:Real};
    phase_offset::Bool = true
) where {N}
    NT = length(TEs)

    N > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    p = similar(phas, size(phas)[1:N-1])

    return if !phase_offset
        fit_echo_linear!(p, phas, W, TEs)
    else
        fit_echo_linear!(p, similar(p), phas, W, TEs)
    end
end

"""
    fit_echo_linear!(
        p::AbstractArray{<:AbstractFloat, N},
        phas::AbstractArray{<:AbstractFloat, M > 1},
        W::AbstractArray{<:AbstractFloat, M > 1},
        TEs::AbstractVector{<:Real}
    ) -> p

Weighted least squares for multi-echo data (phase offset = 0).

### Arguments
- `p::AbstractArray{<:AbstractFloat, N}`: weighted least-squares estimate for phase
- `phas::AbstractArray{<:AbstractFloat, M > 1}`: unwrapped multi-echo phase
- `W::AbstractArray{<:AbstractFloat, M > 1}`: reciprocal of error variance of voxel
- `TEs::AbstractVector{<:Real}`: echo times

### Returns
- `p`: weighted least-squares estimate for phase
"""
function fit_echo_linear!(
    p::AbstractArray{Tp, N},
    phas::AbstractArray{Tphas, M},
    W::AbstractArray{TW, M},
    TEs::AbstractVector{<:Real}
) where {Tp<:AbstractFloat, Tphas<:AbstractFloat, TW<:Real, N, M}
    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    checkshape(axes(p), axes(phas)[1:M-1], (:p, :phas))
    checkshape((NT,), (size(phas, M),), (:TEs, :phas))
    checkshape(W, phas, (:W, :phas))

    vphas = reshape(phas, :, NT)
    vW = reshape(W, :, NT)
    vp = vec(p)

    T = promote_type(Tp, Tphas, TW)
    tes = convert.(T, TEs)

    zeroTp = zero(Tp)

    @batch for I in eachindex(vp)
        w = vW[I,1] * vW[I,1] * tes[1]
        den = w * tes[1]
        num = w * vphas[I,1]
        for t in 2:NT
            w = vW[I,t] * vW[I,t] * tes[t]
            den = muladd(w, tes[t], den)
            num = muladd(w, vphas[I,t], num)
        end
        vp[I] = iszero(den) ? zeroTp : num * inv(den)
    end

    return p
end

"""
    fit_echo_linear!(
        p::AbstractArray{<:AbstractFloat, N},
        p0::AbstractArray{<:AbstractFloat, N},
        phas::AbstractArray{<:AbstractFloat, M > 1},
        W::AbstractArray{<:AbstractFloat, M > 1},
        TEs::AbstractVector{<:Real}
    ) -> (p, p0)

Weighted least squares for multi-echo data (estimate phase offset).

### Arguments
- `p::AbstractArray{<:AbstractFloat, N}`: weighted least-squares estimate for phase
- `p0::AbstractArray{<:AbstractFloat, N}`: weighted least-squares estimate for phase offset
- `phas::AbstractArray{<:AbstractFloat, M > 1}`: unwrapped multi-echo phase
- `W::AbstractArray{<:AbstractFloat, M > 1}`: reciprocal of error variance of voxel
- `TEs::AbstractVector{<:Real}`: echo times

### Returns
- `p`: weighted least-squares estimate for phase
- `p0`: weighted least-squares estimate for phase offset
"""
function fit_echo_linear!(
    p::AbstractArray{Tp, N},
    p0::AbstractArray{Tp0, N},
    phas::AbstractArray{Tphas, M},
    W::AbstractArray{TW, M},
    TEs::AbstractVector{<:Real}
) where {Tp<:AbstractFloat, Tp0<:AbstractFloat, Tphas<:AbstractFloat, TW<:Real, N, M}
    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    checkshape(axes(p), axes(phas)[1:M-1], (:p, :phas))
    checkshape(axes(p0), axes(phas)[1:M-1], (:p0, :phas))
    checkshape((NT,), (size(phas, M),), (:TEs, :phas))
    checkshape(W, phas, (:W, :phas))

    vphas = reshape(phas, :, NT)
    vW = reshape(W, :, NT)
    vp0 = vec(p0)
    vp = vec(p)

    T = promote_type(Tp, Tp0, Tphas, TW)
    tes = convert.(T, TEs)

    zeroTp = zero(Tp)
    zeroTp0 = zero(Tp0)

    @batch for I in eachindex(vp)
        w = vW[I,1]
        x = w * tes[1]
        y = w * vphas[I,1]
        for t in 2:NT
            w += vW[I,t]
            x = muladd(vW[I,t], tes[t], x)
            y = muladd(vW[I,t], vphas[I,t], y)
        end

        if iszero(w)
            vp[I] = zeroTp
            vp0[I] = zeroTp0
            continue
        end

        w = inv(w)
        x *= w
        y *= w

        xx = tes[1] - x
        yy = vphas[I,1] - y
        ww = vW[I,1] * xx
        num = ww * yy
        den = ww * xx
        for t in 2:NT
            xx = tes[t] - x
            yy = vphas[I,t] - y
            ww = vW[I,t] * xx
            num = muladd(ww, yy, num)
            den = muladd(ww, xx, den)
        end

        if !iszero(den)
            vp[I] = num * inv(den)
            vp0[I] = y - vp[I] * x
        else
            vp[I] = zeroTp
            vp0[I] = zeroTp0
        end
    end

    return p, p0
end

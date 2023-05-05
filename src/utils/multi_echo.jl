#####
##### Average
#####

"""
    multi_echo_average(
        phas::AbstractArray{<:AbstractFloat, N > 1};
        TEs::Union{Nothing, AbstractVector{<:Real}} = nothing,
        W::Union{Nothing, AbstractArray{<:AbstractFloat, N}} = nothing,
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing,
    ) -> typeof(similar(phas)){N-1}

(Weighted) Average for multi-echo data.

The weighted average is computed as

```math
\\frac{\\sum_{i=1}^{\\#TEs} TE_i W_i phas_i}{\\sum_{i=1}^{\\#TEs} TE_i W_i}
```

### Arguments
- `phas::AbstractArray{<:AbstractFloat, N > 1}`: unwrapped multi-echo phase

### Keywords
- `TEs::Union{Nothing, AbstractVector{<:Real}} = nothing`: echo times
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, N}} = nothing`: weights
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Returns
- `typeof(similar(phas)){N-1}`: (weighted) average of multi-echo phase
"""
function multi_echo_average(
    phas::AbstractArray{<:AbstractFloat, M};
    TEs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    W::Union{Nothing, AbstractArray{<:AbstractFloat, M}} = nothing,
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
) where {M}
    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))

    p = tfill!(similar(phas, size(phas)[1:M-1]), 0)
    return multi_echo_average!(p, phas, TEs, W, mask)
end

"""
    multi_echo_average!(
        p::AbstractArray{<:AbstractFloat, N-1},
        phas::AbstractArray{<:AbstractFloat, N > 1},
        TEs::Union{Nothing, AbstractVector{<:Real}} = nothing,
        W::Union{Nothing, AbstractArray{<:AbstractFloat, N}} = nothing,
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing,
    ) -> p

(Weighted) Average for multi-echo data.

The weighted average is computed as

```math
\\frac{\\sum_{i=1}^{\\#TEs} TE_i W_i phas_i}{\\sum_{i=1}^{\\#TEs} TE_i W_i}
```

### Arguments
- `p::AbstractArray{<:AbstractFloat, N-1}`: (weighted) average of multi-echo phase
- `phas::AbstractArray{<:AbstractFloat, N > 1}`: unwrapped multi-echo phase
- `TEs::Union{Nothing, AbstractVector{<:Real}} = nothing`: echo times
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, N}} = nothing`: weights
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Returns
- `p`: (weighted) average of multi-echo phase
"""
function multi_echo_average!(
    p::AbstractArray{<:AbstractFloat, N},
    phas::AbstractArray{<:AbstractFloat, M},
    TEs::Union{Nothing, AbstractVector{<:Real}} = nothing,
    W::Union{Nothing, AbstractArray{<:AbstractFloat, M}} = nothing,
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
) where {N, M}
    require_one_based_indexing(p, phas)
    TEs  !== nothing && require_one_based_indexing(TEs)
    W    !== nothing && require_one_based_indexing(W)
    mask !== nothing && require_one_based_indexing(mask)

    NT = TEs !== nothing ? length(TEs) : size(phas, M)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 0 || throw(ArgumentError("data must be multi-echo"))

    checkshape(axes(p), axes(phas)[1:M-1], (:p, :phas))
    checkshape((NT,), (size(phas, M),), (:TEs, :phas))
    W !== nothing && checkshape(W, phas, (:W, :phas))
    mask !== nothing && checkshape(axes(mask), axes(phas)[1:M-1], (:mask, :phas))

    x = reshape(phas, :, NT)
    pp = vec(p)

    if W === nothing && TEs === nothing
        _multi_echo_average!(pp, x, mask)

    elseif W === nothing && TEs !== nothing
        _multi_echo_average!(pp, x, TEs, mask)

    elseif W !== nothing && TEs === nothing
        _multi_echo_average!(pp, x, reshape(W, :, NT), mask)

    elseif W !== nothing && TEs !== nothing
        _multi_echo_average!(pp, x, TEs, reshape(W, :, NT), mask)

    end

    return p
end


function _multi_echo_average!(
    p::AbstractVector,
    x::AbstractMatrix,
    mask::Union{Nothing, AbstractArray},
)
    NT = size(x, 2)
    iNT = inv(NT)
    @batch per=thread for I in eachindex(p)
        if mask === nothing || mask[I]
            p̄ = zero(eltype(x))
            @simd for t in 1:NT
                p̄ += x[I,t]
            end
            p[I] = iNT * p̄
        end
    end
    return p
end

function _multi_echo_average!(
    p::AbstractVector,
    x::AbstractMatrix,
    TEs::AbstractVector,
    mask::Union{Nothing, AbstractArray},
)
    NT = size(x, 2)
    w = Vector{eltype(x)}(TEs)
    w ./= sum(TEs)
    @batch per=thread for I in eachindex(p)
        if mask === nothing || mask[I]
            p̄ = w[1]*x[I,1]
            for t in 2:NT
                p̄ = muladd(w[t], x[I,t], p̄)
            end
            p[I] = p̄
        end
    end
    return p
end

function _multi_echo_average!(
    p::AbstractVector,
    x::AbstractMatrix,
    W::AbstractMatrix,
    mask::Union{Nothing, AbstractArray},
)
    zeroTp = zero(eltype(p))
    NT = size(x, 2)
    @batch per=thread for I in eachindex(p)
        if mask === nothing || mask[I]
            wt = W[I,1]
            sw = wt
            p̄ = wt*x[I,1]
            for t in 2:NT
                wt = W[I,t]
                sw += wt
                p̄ = muladd(wt, x[I,t], p̄)
            end
            p[I] = iszero(sw) ? zeroTp : p̄ / sw
        end
    end
    return p
end

function _multi_echo_average!(
    p::AbstractVector,
    x::AbstractMatrix,
    TEs::AbstractVector,
    W::AbstractMatrix,
    mask::Union{Nothing, AbstractArray},
)
    zeroTp = zero(eltype(p))
    NT = size(x, 2)
    w = Vector{eltype(x)}(TEs)
    @batch per=thread for I in eachindex(p)
        if mask === nothing || mask[I]
            wt = w[1]*W[I,1]
            sw = wt
            p̄ = wt*x[I,1]
            for t in 2:NT
                wt = w[t]*W[I,t]
                sw += wt
                p̄ = muladd(wt, x[I,t], p̄)
            end
            p[I] = iszero(sw) ? zeroTp : p̄ / sw
        end
    end
    return p
end


#####
##### Linear
#####

"""
    multi_echo_linear_fit(
        phas::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real};
        W::Union{Nothing, AbstractArray{<:AbstractFloat, N}} = nothing,
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing,
        phase_offset::Bool = true
    ) -> Tuple{typeof(similar(phas)){N-1}, [typeof(similar(phas)){N-1}]}

(Weighted) Least squares for multi-echo data.

### Arguments
- `phas::AbstractArray{<:AbstractFloat, N > 1}`: unwrapped multi-echo phase
- `TEs::AbstractVector{<:Real}`: echo times

### Keywords
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, N}} = nothing`:
    square root of weights, e.g. reciprocal of error variance of voxel
    ~> W² = 1/σ²(phas) = mag²/σ²(mag)
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest
- `phase_offset::Bool = true`: model phase offset (`true`)

### Returns
- `typeof(similar(phas)){N-1}`: (weighted) least-squares estimate for phase
- [`typeof(similar(phas)){N-1}`]: (weighted) least-squares estimate for phase offset
    if `phase_offset = true`
"""
function multi_echo_linear_fit(
    phas::AbstractArray{<:AbstractFloat, M},
    TEs::AbstractVector{<:Real};
    W::Union{Nothing, AbstractArray{<:AbstractFloat, M}} = nothing,
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    phase_offset::Bool = true
) where {M}
    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > Int(phase_offset) || throw(ArgumentError("data must be multi-echo"))

    p = tfill!(similar(phas, size(phas)[1:M-1]), 0)

    return if !phase_offset
        multi_echo_linear_fit!(p, phas, TEs, W, mask)
    else
        multi_echo_linear_fit!(p, tfill!(similar(p), 0), phas, TEs, W, mask)
    end
end

"""
    multi_echo_linear_fit!(
        p::AbstractArray{<:AbstractFloat, N-1},
        phas::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        W::Union{Nothing, AbstractArray{<:AbstractFloat, N}},
        mask::Union{Nothing, AbstractArray{Bool, N-1}},
    ) -> p

(Weighted) Least squares for multi-echo data (phase offset = 0).

### Arguments
- `p::AbstractArray{<:AbstractFloat, N-1}`: (weighted) least-squares estimate for phase
- `phas::AbstractArray{<:AbstractFloat, N > 1}`: unwrapped multi-echo phase
- `TEs::AbstractVector{<:Real}`: echo times
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, N}}`:
    square root of weights, e.g. reciprocal of error variance of voxel
    ~> W² = 1/σ²(phas) = mag²/σ²(mag)
- `mask::Union{Nothing, AbstractArray{Bool, N-1}}`:
    binary mask of region of interest

### Returns
- `p`: (weighted) least-squares estimate for phase
"""
function multi_echo_linear_fit!(
    p::AbstractArray{<:AbstractFloat, N},
    phas::AbstractArray{<:AbstractFloat, M},
    TEs::AbstractVector{<:Real},
    W::Union{Nothing, AbstractArray{<:AbstractFloat, M}},
    mask::Union{Nothing, AbstractArray{Bool}},
) where {N, M}
    require_one_based_indexing(p, phas, TEs)
    W !== nothing && require_one_based_indexing(W)
    mask !== nothing && require_one_based_indexing(mask)

    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 0 || throw(ArgumentError("data must be multi-echo"))

    checkshape(axes(p), axes(phas)[1:M-1], (:p, :phas))
    checkshape((NT,), (size(phas, M),), (:TEs, :phas))
    W !== nothing && checkshape(W, phas, (:W, :phas))
    mask !== nothing && checkshape(axes(mask), axes(phas)[1:M-1], (:mask, :phas))

    X = TEs
    Y = reshape(phas, :, NT)
    β = vec(p)

    if W === nothing
        _multi_echo_linear_fit!(β, Y, X, mask)
    else
        _multi_echo_linear_fit!(β, Y, X, reshape(W, :, NT), mask)
    end

    return p
end

function _multi_echo_linear_fit!(
    β::AbstractVector,
    Y::AbstractMatrix,
    X::AbstractVector,
    mask::Union{Nothing, AbstractArray},
)
    zeroβ = zero(eltype(β))
    sxx = sum(abs2, X)

    if iszero(sxx)
        @batch for I in eachindex(β)
            if mask === nothing || mask[I]
                β[I] = zeroβ
            end
        end

    else
        NT = size(Y, 2)
        x = Vector{eltype(Y)}(X)
        x ./= sxx

        @batch per=thread for I in eachindex(β)
            if mask === nothing || mask[I]
                sxy = x[1] * Y[I,1]
                for t in 2:NT
                    sxy = muladd(x[t], Y[I,t], sxy)
                end
                β[I] = sxy
            end
        end
    end

    return β
end

function _multi_echo_linear_fit!(
    β::AbstractVector,
    Y::AbstractMatrix,
    X::AbstractVector,
    W::AbstractMatrix,
    mask::Union{Nothing, AbstractArray},
)
    zeroβ = zero(eltype(β))
    NT = size(Y, 2)

    x = Vector{eltype(W)}(X)

    @batch per=thread for I in eachindex(β)
        if mask === nothing || mask[I]
            xt, wt, yt = x[1], W[I,1], Y[I,1]
            xw2 = xt * wt * wt
            sxy = xw2 * yt
            sxx = xw2 * xt
            for t in 2:NT
                xt, wt, yt = x[t], W[I,t], Y[I,t]
                xw2 = xt * wt * wt
                sxy = muladd(xw2, yt, sxy)
                sxx = muladd(xw2, xt, sxx)
            end
            β[I] = iszero(sxx) ? zeroβ : sxy / sxx
        end
    end

    return β
end

"""
    multi_echo_linear_fit!(
        p::AbstractArray{<:AbstractFloat, N-1},
        p0::AbstractArray{<:AbstractFloat, N-1},
        phas::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        W::Union{Nothing, AbstractArray{<:AbstractFloat, N}},
        mask::Union{Nothing, AbstractArray{Bool, N-1}},
    ) -> (p, p0)

(Weighted) Least squares for multi-echo data (estimate phase offset).

### Arguments
- `p::AbstractArray{<:AbstractFloat, N-1}`: (weighted) least-squares estimate for phase
- `p0::AbstractArray{<:AbstractFloat, N-1}`: (weighted) least-squares estimate for phase offset
- `phas::AbstractArray{<:AbstractFloat, N > 1}`: unwrapped multi-echo phase
- `TEs::AbstractVector{<:Real}`: echo times
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, N}}`:
    square root of weights, e.g. reciprocal of error variance of voxel
    ~> W² = 1/σ²(phas) = mag²/σ²(mag)
- `mask::Union{Nothing, AbstractArray{Bool, N-1}}`:
    binary mask of region of interest

### Returns
- `p`: (weighted) least-squares estimate for phase
- `p0`: (weighted) least-squares estimate for phase offset
"""
function multi_echo_linear_fit!(
    p::AbstractArray{<:AbstractFloat, N},
    p0::AbstractArray{<:AbstractFloat, N},
    phas::AbstractArray{<:AbstractFloat, M},
    TEs::AbstractVector{<:Real},
    W::Union{Nothing, AbstractArray{<:AbstractFloat, M}},
    mask::Union{Nothing, AbstractArray{Bool}},
) where {N, M}
    require_one_based_indexing(p, p0, phas, TEs)
    W !== nothing && require_one_based_indexing(W)
    mask !== nothing && require_one_based_indexing(mask)

    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    checkshape(axes(p), axes(phas)[1:M-1], (:p, :phas))
    checkshape(axes(p0), axes(phas)[1:M-1], (:p0, :phas))
    checkshape((NT,), (size(phas, M),), (:TEs, :phas))
    W !== nothing && checkshape(W, phas, (:W, :phas))
    mask !== nothing && checkshape(axes(mask), axes(phas)[1:M-1], (:mask, :phas))

    X = TEs
    Y = reshape(phas, :, NT)
    β = vec(p)
    α = vec(p0)

    if W === nothing
        _multi_echo_linear_fit!(β, α, Y, X, mask)
    else
        _multi_echo_linear_fit!(β, α, Y, X, reshape(W, :, NT), mask)
    end

    return p, p0
end

function _multi_echo_linear_fit!(
    β::AbstractVector,
    α::AbstractVector,
    Y::AbstractMatrix,
    X::AbstractVector,
    mask::Union{Nothing, AbstractArray},
)
    zeroβ = zero(eltype(β))
    NT = size(Y, 2)
    iNT = inv(NT)

    x̄ = sum(X) * iNT
    xx̄ = Vector{eltype(Y)}(X)
    xx̄ .-= x̄
    s2x  = sum(abs2, xx̄)

    if iszero(s2x)
        @batch per=thread for I in eachindex(β)
            if mask === nothing || mask[I]
                ȳ = zero(eltype(Y))
                @simd for t in 1:NT
                    ȳ += Y[I,t]
                end
                α[I] = iNT * ȳ
                β[I] = zeroβ
            end
        end

    else
        xx̄ ./= s2x

        @batch per=thread for I in eachindex(β)
            if mask === nothing || mask[I]
                ȳ = zero(eltype(Y))
                @simd for t in 1:NT
                    ȳ += Y[I,t]
                end
                ȳ *= iNT

                sxy = xx̄[1] * (Y[I,1] - ȳ)
                for t in 2:NT
                    sxy = muladd(xx̄[t], (Y[I,t] - ȳ), sxy)
                end

                β[I] = sxy
                α[I] = ȳ - β[I]*x̄
            end
        end
    end

    return β, α
end

function _multi_echo_linear_fit!(
    β::AbstractVector,
    α::AbstractVector,
    Y::AbstractMatrix,
    X::AbstractVector,
    W::AbstractMatrix,
    mask::Union{Nothing, AbstractArray},
)
    zeroβ = zero(eltype(β))
    zeroα = zero(eltype(α))
    NT = size(Y, 2)

    x = Vector{eltype(W)}(X)

    @batch per=thread for I in eachindex(β)
        if mask === nothing || mask[I]
            wt = W[I,1]
            w = wt
            x̄ = wt * x[1]
            ȳ = wt * Y[I,1]
            for t in 2:NT
                wt = W[I,t]
                w += wt
                x̄ = muladd(wt, x[t], x̄)
                ȳ = muladd(wt, Y[I,t], ȳ)
            end

            if iszero(w)
                β[I] = zeroβ
                α[I] = zeroα
                continue
            end

            x̄ /= w
            ȳ /= w

            xx̄ = x[1] - x̄
            yȳ = Y[I,1] - ȳ

            wxx̄ = W[I,1] * xx̄

            sxy = wxx̄ * yȳ
            s2x = wxx̄ * xx̄

            for t in 2:NT
                xx̄ = x[t] - x̄
                yȳ = Y[I,t] - ȳ

                wxx̄ = W[I,t] * xx̄

                sxy = muladd(wxx̄, yȳ, sxy)
                s2x = muladd(wxx̄, xx̄, s2x)
            end

            β[I] = iszero(s2x) ? zeroβ : sxy / s2x
            α[I] = ȳ - β[I]*x̄
        end
    end

    return β, α
end

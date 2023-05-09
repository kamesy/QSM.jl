"""
    r2star_ll(
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing
    ) -> typeof(similar(mag, size(mag)[1:N-1]))

Log-linear fit.

### Arguments
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Returns
- `typeof(similar(mag, size(mag)[1:N-1]))`: R2* map (1 / units of TEs)
"""
function r2star_ll(
    mag::AbstractArray{<:AbstractFloat, M},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {M}
    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    r2s = tfill!(similar(mag, size(mag)[1:M-1]), 0)
    return r2star_ll!(r2s, mag, TEs, mask)
end

"""
    r2star_ll!(
        r2s::AbstractArray{<:AbstractFloat, N-1},
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing
    ) -> r2s

Log-linear fit.

### Arguments
- `r2s::AbstractArray{<:AbstractFloat, N-1}`: R2* map (1 / units of TEs)
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Returns
- `r2s`: R2* map (1 / units of TEs)
"""
function r2star_ll!(
    r2s::AbstractArray{<:AbstractFloat, N},
    mag::AbstractArray{T, M},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {T<:AbstractFloat, N, M}
    require_one_based_indexing(r2s, mag, TEs)
    mask !== nothing && require_one_based_indexing(mask)

    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    checkshape(axes(r2s), axes(mag)[1:M-1], (:r2s, :mag))
    checkshape((NT,), (size(mag, M),), (:TEs, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:M-1], (:mask, :mag))

    iNT = inv(NT)

    X = -TEs
    Y = reshape(mag, :, NT)

    x̄ = sum(X) * iNT
    xx̄ = collect(X .- x̄)
    s2x = sum(abs2, xx̄)

    if iszero(s2x)
        @batch for I in eachindex(r2s)
            if mask === nothing || mask[I]
                r2s[I] = 0
            end
        end

    else
        xx̄ ./= s2x

        @batch per=thread threadlocal=zeros(T, NT)::Vector{T} for I in eachindex(r2s)
            if mask === nothing || mask[I]
                logy = threadlocal

                ly = log(Y[I,1])
                ȳ = ly
                logy[1] = ly
                for t in 2:NT
                    ly = log(Y[I,t])
                    ȳ += ly
                    logy[t] = ly
                end
                ȳ *= iNT

                sxy = xx̄[1] * (logy[1] - ȳ)
                for t in 2:NT
                    sxy = muladd(xx̄[t], (logy[t] - ȳ), sxy)
                end

                r2s[I] = sxy
            end
        end
    end

    return r2s
end


"""
    r2star_arlo(
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing
    ) -> typeof(similar(mag, size(mag)[1:N-1]))

Auto-Regression on Linear Operations (ARLO) [1].

### Arguments
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Returns
- `typeof(similar(mag, size(mag)[1:N-1]))`: R2* map (1 / units of TEs)

### References
[1] Pei M, Nguyen TD, Thimmappa ND, Salustri C, Dong F, Cooper MA, Li J,
    Prince MR, Wang Y. Algorithm for fast monoexponential fitting based on
    auto‐regression on linear operations (ARLO) of data.
    Magnetic resonance in medicine. 2015 Feb;73(2):843-50.
"""
function r2star_arlo(
    mag::AbstractArray{<:AbstractFloat, M},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {M}
    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 2 || throw(ArgumentError("ARLO requires at least 3 echoes"))

    r2s = tfill!(similar(mag, size(mag)[1:M-1]), 0)
    return r2star_arlo!(r2s, mag, TEs, mask)
end

"""
    r2star_arlo!(
        r2s::AbstractArray{<:AbstractFloat, N-1},
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing
    ) -> r2s

Auto-Regression on Linear Operations (ARLO) [1].

### Arguments
- `r2s::AbstractArray{<:AbstractFloat, N-1}`: R2* map (1 / units of TEs)
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Returns
- `r2s`: R2* map (1 / units of TEs)

### References
[1] Pei M, Nguyen TD, Thimmappa ND, Salustri C, Dong F, Cooper MA, Li J,
    Prince MR, Wang Y. Algorithm for fast monoexponential fitting based on
    auto‐regression on linear operations (ARLO) of data.
    Magnetic resonance in medicine. 2015 Feb;73(2):843-50.
"""
function r2star_arlo!(
    r2s::AbstractArray{<:AbstractFloat, N},
    mag::AbstractArray{<:AbstractFloat, M},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {N, M}
    require_one_based_indexing(r2s, mag, TEs)
    mask !== nothing && require_one_based_indexing(mask)

    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 2 || throw(ArgumentError("ARLO requires at least 3 echoes"))

    checkshape(axes(r2s), axes(mag)[1:M-1], (:r2s, :mag))
    checkshape((NT,), (size(mag, M),), (:TEs, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:M-1], (:mask, :mag))

    all(≈(TEs[2]-TEs[1]), diff(TEs)) ||
        throw(DomainError("ARLO requires equidistant echoes"))

    zeroT = zero(eltype(r2s))
    α = convert(eltype(mag), 3 / (TEs[2]-TEs[1]))

    P = reshape(mag, :, NT)

    @batch per=thread for I in eachindex(r2s)
        if mask === nothing || mask[I]
            m0, m1, m2 = P[I,1], P[I,2], P[I,3]

            δ = m0 - m2
            s = m0 + muladd(4, m1, m2)
            a = muladd(2, m1, m0)

            num = δ * a
            den = s * a

            for t in 2:NT-2
                m0 = m1
                m1 = m2
                m2 = P[I,t+2]

                δ = m0 - m2
                s = m0 + muladd(4, m1, m2)
                a = muladd(2, m1, m0)

                num = muladd(δ, a, num)
                den = muladd(s, a, den)
            end

            r2s[I] = iszero(den) ? zeroT : α * num / den
        end
    end

    return r2s
end


"""
    r2star_crsi(
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing;
        M::Integer = 3,
        sigma::Union{Nothing, Real} = nothing,
        Rsz::NTuple{N-1, Integer} = size(mag)[1:N-1] .÷ 20,
    ) -> typeof(similar(mag, size(mag)[1:N-1]))

Calculation of Relaxivities by Signal Integration (CRSI) [1].

### Arguments
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Keywords
- `M::Integer = 3`: interpolation factor
- `sigma::Union{Nothing, Real} = nothing`: noise
- `Rsz::NTuple{N-1, Integer} = size(mag)[1:N-1] .÷ 20`:
    - `sigma isa Real`: unused
    - `sigma isa Nothing`: size of kernels used to calculate the noise from the
        background signal of the magnitude.

### Returns
- `typeof(similar(mag, size(mag)[1:N-1]))`: R2* map (1 / units of TEs)

### References
[1] Song R, Loeffler RB, Holtrop JL, McCarville MB, Hankins JS, Hillenbrand CM.
    Fast quantitative parameter maps without fitting: Integration yields
    accurate mono‐exponential signal decay rates.
    Magnetic resonance in medicine. 2018 Jun;79(6):2978-85.
"""
function r2star_crsi(
    mag::AbstractArray{<:AbstractFloat, NM},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing;
    M::Integer = 3,
    sigma::Union{Nothing, Real} = nothing,
    Rsz::NTuple{NR, Integer} = size(mag)[1:NM-1] .÷ 20,
) where {NM, NR}
    NT = length(TEs)

    NM > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))
    M  > 0 || throw(ArgumentError("interpolation factor M must be greater than 0"))

    r2s = tfill!(similar(mag, size(mag)[1:NM-1]), 0)
    return r2star_crsi!(r2s, mag, TEs, mask, M, sigma, Rsz)
end

"""
    r2star_crsi!(
        r2s::AbstractArray{<:AbstractFloat, N-1},
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing,
        M::Integer = 3,
        sigma::Union{Nothing, Real} = nothing,
        Rsz::NTuple{N-1, Integer} = size(mag)[1:N-1] .÷ 20,
    ) -> r2s

Calculation of Relaxivities by Signal Integration (CRSI) [1].

### Arguments
- `r2s::AbstractArray{<:AbstractFloat, N-1}`: R2* map (1 / units of TEs)
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest
- `M::Integer = 3`: interpolation factor
- `sigma::Union{Nothing, Real} = nothing`: noise
- `Rsz::NTuple{N-1, Integer} = size(mag)[1:N-1] .÷ 20`:
    - `sigma isa Real`: unused
    - `sigma isa Nothing`: size of kernels used to calculate the noise from the
        background signal of the magnitude.

### Returns
- `r2s`: R2* map (1 / units of TEs)

### References
[1] Song R, Loeffler RB, Holtrop JL, McCarville MB, Hankins JS, Hillenbrand CM.
    Fast quantitative parameter maps without fitting: Integration yields
    accurate mono‐exponential signal decay rates.
    Magnetic resonance in medicine. 2018 Jun;79(6):2978-85.
"""
function r2star_crsi!(
    r2s::AbstractArray{<:AbstractFloat, N},
    mag::AbstractArray{T, NM},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing,
    M::Integer = 3,
    sigma::Union{Nothing, Real} = nothing,
    Rsz::NTuple{NR, Integer} = size(mag)[1:NM-1] .÷ 20,
) where {T<:AbstractFloat, N, NM, NR}
    require_one_based_indexing(r2s, mag, TEs)
    mask !== nothing && require_one_based_indexing(mask)

    NT = length(TEs)

    NM > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))
    M  > 0 || throw(ArgumentError("interpolation factor M must be greater than 0"))

    checkshape(axes(r2s), axes(mag)[1:NM-1], (:r2s, :mag))
    checkshape((NT,), (size(mag, NM),), (:TEs, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:NM-1], (:mask, :mag))
    sigma === nothing && checkshape((NR,), (NM-1,), (:Rsz, :mag))

    zeroT = zero(eltype(r2s))

    τ  = Vector{T}(diff(TEs) ./ (M+1))
    γ0 = Vector{T}([(2*M - 2*m + 1) / (2*M + 2) for m in 0:M])
    γ1 = Vector{T}([(2*m + 1) / (2*M + 2) for m in 0:M])

    σ2 = sigma === nothing ? _noise_crsi(mag, Rsz, mask) : (2*sigma*sigma)
    d0 = convert(T, -σ2 * (TEs[end]-TEs[1]) / 2)

    P = reshape(mag, :, NT)

    @batch per=thread for I in eachindex(r2s)
        if mask === nothing || mask[I]
            p1  = P[I,1] * P[I,1]
            num = p1
            den = d0
            for t in 1:NT-1
                p0 = p1
                p1 = P[I,t+1] * P[I,t+1]
                p  = pow(p0, γ0[1]) * pow(p1, γ1[1])
                for m in 2:M+1
                    p = muladd(pow(p0, γ0[m]), pow(p1, γ1[m]), p)
                end
                den = muladd(τ[t], p, den)
            end
            num -= p1
            r2s[I] = iszero(den) ? zeroT : num / (2*den)
        end
    end

    return r2s
end

function _noise_crsi(
    mag::AbstractArray{T, N},
    Rsz::NTuple{M, Integer},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {T<:AbstractFloat, N, M}
    require_one_based_indexing(mag)
    mask !== nothing && require_one_based_indexing(mask)

    checkshape((M,), (N-1,), (:Rsz, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:3], (:mask, :mag))

    sz = size(mag)[1:M]
    rsz = map(min, Rsz, (sz.-2).÷2)

    outer = ntuple(d -> 2:sz[d]-1, Val(M))
    inner = ntuple(d -> 2+rsz[d]:sz[d]-1-rsz[d], Val(M))

    m = Ref(0)
    s = Ref(zero(T))

    for t in axes(mag, N)
        _edgeloop(outer, inner) do I...
            if all(map(∉, I, inner)) && (mask === nothing || (@inbounds !mask[I...]))
                p = @inbounds mag[I...,t]
                s[] += p*p
                m[] += 1
            end
        end
    end

    σ2 = s[]
    n = m[]

    return iszero(n) ? zero(T) : σ2 / n
end


"""
    r2star_numart2s(
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing
    ) -> typeof(similar(mag, size(mag)[1:N-1]))

Numerical Algorithm for Real-time T2* mapping (NumART2*) [1].

### Arguments
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Returns
- `typeof(similar(mag, size(mag)[1:N-1]))`: R2* map (1 / units of TEs)

### References
[1] Hagberg GE, Indovina I, Sanes JN, Posse S. Real‐time quantification of T2*
    changes using multiecho planar imaging and numerical methods.
    Magnetic Resonance in Medicine: An Official Journal of the International
    Society for Magnetic Resonance in Medicine. 2002 Nov;48(5):877-82.
"""
function r2star_numart2s(
    mag::AbstractArray{<:AbstractFloat, M},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {M}
    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    r2s = tfill!(similar(mag, size(mag)[1:M-1]), 0)
    return r2star_numart2s!(r2s, mag, TEs, mask)
end

"""
    r2star_numart2s!(
        r2s::AbstractArray{<:AbstractFloat, N-1},
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing
    ) -> r2s

Numerical Algorithm for Real-time T2* mapping (NumART2*) [1].

### Arguments
- `r2s::AbstractArray{<:AbstractFloat, N-1}`: R2* map (1 / units of TEs)
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool, N-1}} = nothing`:
    binary mask of region of interest

### Returns
- `r2s`: R2* map (1 / units of TEs)

### References
[1] Hagberg GE, Indovina I, Sanes JN, Posse S. Real‐time quantification of T2*
    changes using multiecho planar imaging and numerical methods.
    Magnetic Resonance in Medicine: An Official Journal of the International
    Society for Magnetic Resonance in Medicine. 2002 Nov;48(5):877-82.
"""
function r2star_numart2s!(
    r2s::AbstractArray{<:AbstractFloat, N},
    mag::AbstractArray{<:AbstractFloat, M},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {N, M}
    require_one_based_indexing(r2s, mag, TEs)
    mask !== nothing && require_one_based_indexing(mask)

    NT = length(TEs)

    M > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    checkshape(axes(r2s), axes(mag)[1:M-1], (:r2s, :mag))
    checkshape((NT,), (size(mag, M),), (:TEs, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:M-1], (:mask, :mag))

    zeroT = zero(eltype(r2s))
    α = convert(eltype(mag), 2*(NT-1) / (TEs[end]-TEs[1]))

    P = reshape(mag, :, NT)

    @batch per=thread for I in eachindex(r2s)
        if mask === nothing || mask[I]
            m = P[I,1]
            β = m
            γ = m
            for t in 2:NT-1
                m = P[I,t]
                γ += m + m
            end
            m  = P[I,end]
            β -= m
            γ += m
            r2s[I] = iszero(γ) ? zeroT : α * β / γ
        end
    end

    return r2s
end

"""
    function r2star_ll(
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool}} = nothing
    ) -> typeof(similar(mag, size(mag)[1:N-1]))

Log-linear fit.

### Arguments
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool}} = nothing`: binary mask of region of interest

### Returns
- `typeof(similar(mag, size(mag)[1:N-1]))`: R2* map (1 / units of TEs)
"""
function r2star_ll(
    mag::AbstractArray{T, N},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {T<:AbstractFloat, N}
    NT = length(TEs)

    N > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    checkshape((NT,), (size(mag, N),), (:TEs, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:N-1], (:mask, :mag))

    r2s = similar(mag, size(mag)[1:N-1])
    r2s = tfill!(r2s, zero(T))

    vmag = reshape(mag, :, NT)
    vr2s = vec(r2s)

    A = qr(Matrix{T}([-[TEs...] ones(NT)]))

    if mask === nothing
        b = tmap(log, transpose(vmag))
        x̂ = ldiv!(A, b)
        @batch for I in eachindex(vr2s)
            vr2s[I] = x̂[1,I]
        end

    else
        i = 0
        R = Vector{Int}(undef, sum(mask))
        @inbounds for I in eachindex(mask)
            if mask[I]
                R[i += 1] = I
            end
        end

        b = similar(mag, (NT, length(R)))
        for t in 1:NT
            bt = @view(b[t,:])
            mt = @view(vmag[:,t])
            @batch for I in eachindex(R)
                bt[I] = log(mt[R[I]])
            end
        end

        x̂ = ldiv!(A, b)

        @batch for I in eachindex(R)
            vr2s[R[I]] = x̂[1,I]
        end
    end

    return r2s
end


"""
    function r2star_arlo(
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool}} = nothing
    ) -> typeof(similar(mag, size(mag)[1:N-1]))

Auto-Regression on Linear Operations (ARLO) [1].

### Arguments
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool}} = nothing`: binary mask of region of interest

### Returns
- `typeof(similar(mag, size(mag)[1:N-1]))`: R2* map (1 / units of TEs)

### References
[1] Pei M, Nguyen TD, Thimmappa ND, Salustri C, Dong F, Cooper MA, Li J,
    Prince MR, Wang Y. Algorithm for fast monoexponential fitting based on
    auto‐regression on linear operations (ARLO) of data.
    Magnetic resonance in medicine. 2015 Feb;73(2):843-50.
"""
function r2star_arlo(
    mag::AbstractArray{T, N},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {T<:AbstractFloat, N}
    NT = length(TEs)

    N > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 2 || throw(ArgumentError("ARLO requires at least 3 echoes"))

    checkshape((NT,), (size(mag, N),), (:TEs, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:N-1], (:mask, :mag))

    all((≈)(TEs[2]-TEs[1]), TEs[2:end].-TEs[1:end-1]) ||
        throw(DomainError("ARLO requires equidistant echoes"))

    r2s = similar(mag, size(mag)[1:N-1])
    r2s = tfill!(r2s, zero(T))

    vmag = reshape(mag, :, NT)
    vr2s = vec(r2s)

    zeroT = zero(T)
    twoT  = convert(T, 2)
    fourT = convert(T, 4)

    α = convert(T, 3 / (TEs[2]-TEs[1]))

    @batch for I in eachindex(vr2s)
        if mask === nothing || mask[I]
            m0 = vmag[I,1]
            m1 = vmag[I,2]
            m2 = vmag[I,3]

            δ = m0 - m2
            s = m0 + muladd(fourT, m1, m2)
            a = muladd(twoT, m1, m0)

            num = δ * a
            den = s * a

            for t in 2:NT-2
                m0 = m1
                m1 = m2
                m2 = vmag[I,t+2]

                δ = m0 - m2
                s = m0 + muladd(fourT, m1, m2)
                a = muladd(twoT, m1, m0)

                num = muladd(δ, a, num)
                den = muladd(s, a, den)
            end

            vr2s[I] = iszero(den) ? zeroT : α * num * inv(den)
        end
    end

    return r2s
end


"""
    function r2star_crsi(
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool}} = nothing;
        M::Integer = 3,
        sigma::Union{Nothing, Real} = nothing,
        Rsz::NTuple{N-1, Integer} = size(mag)[1:N-1] .÷ 20,
    ) -> typeof(similar(mag, size(mag)[1:N-1]))

Calculation of Relaxivities by Signal Integration (CRSI) [1].

### Arguments
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool}} = nothing`: binary mask of region of interest

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
    mag::AbstractArray{T, N},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing;
    M::Integer = 3,
    sigma::Union{Nothing, Real} = nothing,
    Rsz::NTuple{NR, Integer} = size(mag)[1:N-1] .÷ 20,
) where {T<:AbstractFloat, N, NR}
    NT = length(TEs)

    N > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))
    M > 0 || throw(ArgumentError("interpolation factor M must be greater than 0"))

    checkshape((NT,), (size(mag, N),), (:TEs, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:N-1], (:mask, :mag))
    sigma !== nothing && checkshape((NR,), (N-1,), (:Rsz, :mag))

    P = tmap(x -> x*x, mag)
    r2s = similar(mag, size(mag)[1:N-1])
    r2s = tfill!(r2s, zero(T))

    vP = reshape(P, :, NT)
    vr2s = vec(r2s)

    if sigma !== nothing
        σ2 = convert(T, 2*sigma*sigma)
    else
        σ2 = _noise_crsi(P, Rsz, mask)
    end

    zeroT = zero(T)
    α = convert(T, 1//2)
    β = convert(T, -α * σ2 * (TEs[end] - TEs[1]))

    τ  = SVector{NT-1, T}((TEs[2:end] .- TEs[1:end-1]) ./ (M+1))
    γ0 = SVector{M+1, T}([(2*M - 2*m + 1) / (2*M + 2) for m in 0:M]...)
    γ1 = SVector{M+1, T}([(2*m + 1) / (2*M + 2) for m in 0:M]...)

    @batch for I in eachindex(vr2s)
        if mask === nothing || mask[I]
            den = β
            for t in 1:NT-1
                p0 = vP[I,t]
                p1 = vP[I,t+1]
                p = pow(p0, γ0[1]) * pow(p1, γ1[1])
                for m in 2:M+1
                    p = muladd(pow(p0, γ0[m]), pow(p1, γ1[m]), p)
                end
                den = muladd(τ[t], p, den)
            end

            vr2s[I] = iszero(den) ? zeroT : α * (vP[I,1] - vP[I,end]) * inv(den)
        end
    end

    return r2s
end

function _noise_crsi(
    P::AbstractArray{T, N},
    Rsz::NTuple{M, Integer},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {T<:AbstractFloat, N, M}
    checkshape((M,), (N-1,), (:Rsz, :P))
    mask !== nothing && checkshape(axes(mask), axes(P)[1:3], (:mask, :P))

    sz = size(P)[1:M]
    rsz = map(min, Rsz, (sz.-2).÷2)

    outer = ntuple(d -> 2:sz[d]-1, Val(M))
    inner = ntuple(d -> 2+rsz[d]:sz[d]-1-rsz[d], Val(M))

    m = Ref(0)
    s = Ref(zero(T))

    for t in axes(P, N)
        _edgeloop(outer, inner) do I...
            if all(map(∉, I, inner)) && (mask === nothing || (@inbounds !mask[I...]))
                s[] += (@inbounds P[I...,t])
                m[] += 1
            end
        end
    end

    σ2 = s[]
    n = m[]

    return iszero(n) ? zero(T) : σ2 / n
end


"""
    function r2star_numart2s(
        mag::AbstractArray{<:AbstractFloat, N > 1},
        TEs::AbstractVector{<:Real},
        mask::Union{Nothing, AbstractArray{Bool}} = nothing
    ) -> typeof(similar(mag, size(mag)[1:N-1]))

Numerical Algorithm for Real-time T2* mapping (NumART2*) [1].

### Arguments
- `mag::AbstractArray{<:AbstractFloat, N > 1}`: multi-echo magnitude
- `TEs::AbstractVector{<:Real}`: echo times
- `mask::Union{Nothing, AbstractArray{Bool}} = nothing`: binary mask of region of interest

### Returns
- `typeof(similar(mag, size(mag)[1:N-1]))`: R2* map (1 / units of TEs)

### References
[1] Hagberg GE, Indovina I, Sanes JN, Posse S. Real‐time quantification of T2*
    changes using multiecho planar imaging and numerical methods.
    Magnetic Resonance in Medicine: An Official Journal of the International
    Society for Magnetic Resonance in Medicine. 2002 Nov;48(5):877-82.
"""
function r2star_numart2s(
    mag::AbstractArray{T, N},
    TEs::AbstractVector{<:Real},
    mask::Union{Nothing, AbstractArray{Bool}} = nothing
) where {T<:AbstractFloat, N}
    NT = length(TEs)

    N > 1 || throw(ArgumentError("array must contain echoes in last dimension"))
    NT > 1 || throw(ArgumentError("data must be multi-echo"))

    checkshape((NT,), (size(mag, N),), (:TEs, :mag))
    mask !== nothing && checkshape(axes(mask), axes(mag)[1:N-1], (:mask, :mag))

    r2s = similar(mag, size(mag)[1:N-1])
    r2s = tfill!(r2s, zero(T))

    vmag = reshape(mag, :, NT)
    vr2s = vec(r2s)

    zeroT = zero(T)
    α = convert(T, 2*(NT - 1) / (TEs[end] - TEs[1]))

    @batch for I in eachindex(vr2s)
        if mask === nothing || mask[I]
            den = vmag[I,1]
            for t in 2:NT-1
                den += vmag[I,t] + vmag[I,t]
            end
            den += vmag[I,NT]
            vr2s[I] = iszero(den) ? zeroT : α * (vmag[I,1] - vmag[I,NT]) * inv(den)
        end
    end

    return r2s
end

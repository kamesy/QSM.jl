"""
    homodyne(
        u::AbstractArray{<:Complex{<:AbstractFloat}, N > 1},
        wsz::Union{Integer, NTuple{2, Integer}},
        window::Symbol,
        args...
    ) -> typeof(similar(u))

    homodyne!(
        u::AbstractArray{<:Complex{<:AbstractFloat}, N > 1},
        wsz::Union{Integer, NTuple{2, Integer}},
        window::Symbol,
        args...
    ) -> u

    homodyne!(
        v::AbstractArray{<:Complex{<:AbstractFloat}, N > 1},
        u::AbstractArray{<:Complex{<:AbstractFloat}, N},
        wsz::Union{Integer, NTuple{2, Integer}},
        window::Symbol,
        args...
    ) -> v

Homodyne filter.

Filter by applying 2d window to k-space of complex data: `u / F^-1(w * F(u))`.

### Arguments
- `u::AbstractArray{<:Complex{<:AbstractFloat}, N > 1}`:
    complex (multi-echo) data
- `wsz::Union{Integer, NTuple{2, Integer}}`: window size
- `window::Symbol`: window function
    - `:fermi`: `w = 1 / (1 + exp((x-radius) / width)), x ∈ [-0.5, 0.5]`
    - `:gaussian`: `w = exp(-0.5*(x/σ)^2), x ∈ [-0.5, 0.5]`
    - `:hamming`: `w = 0.54 + 0.46*cos(2π*x), x ∈ [-0.5, 0.5]`
    - `:hann`: `w = 0.5*(1 + cos(2π*x)), x ∈ [-0.5, 0.5]`
- `args...`:
    - `window == :fermi`: `radius::Real, width::Real`
    - `window == :gaussian`: `σ::Real`
    - otherwise: unused

### Returns
- `typeof(similar(u)) / u / v`: homodyne filtered complex (multi-echo) data
"""
function homodyne(
    u::AbstractArray{<:Complex{<:AbstractFloat}, N},
    wsz::Union{Integer, NTuple{2, Integer}},
    window::Symbol,
    args...
) where {N}
    N > 1 || throw(ArgumentError("array must be at least 2d"))
    return homodyne!(similar(u), u, wsz, window, args...)
end

function homodyne!(
    u::AbstractArray{<:Complex{<:AbstractFloat}, N},
    wsz::Union{Integer, NTuple{2, Integer}},
    window::Symbol,
    args...
) where {N}
    N > 1 || throw(ArgumentError("array must be at least 2d"))
    return homodyne!(u, tcopy(u), wsz, window, args...)
end

function homodyne!(
    v::AbstractArray{<:Complex{<:AbstractFloat}, N},
    u::AbstractArray{<:Complex{<:AbstractFloat}, N},
    wsz::Union{Integer, NTuple{2, Integer}},
    window::Symbol,
    args...
) where {N}
    N > 1 || throw(ArgumentError("arrays must be at least 2d"))
    w = makewindow(size(u)[1:2], wsz, window, args...)
    return homodyne!(v, u, w)
end

"""
    homodyne!(
        v::AbstractArray{<:Complex{<:AbstractFloat}, N},
        u::AbstractArray{<:Complex{<:AbstractFloat}, N},
        w::AbstractArray{<:AbstractFloat, M <= N},
    ) -> v

Homodyne filter.

Filter by applying Md window to k-space of complex data: `v = u / F^-1(w * F(u))`.

### Arguments
- `v::AbstractArray{<:Complex{<:AbstractFloat}, N}`:
    homodyne filtered complex (multi-echo) data
- `u::AbstractArray{<:Complex{<:AbstractFloat}, N}`:
    complex (multi-echo) data
- `w::AbstractArray{<:AbstractFloat, M <= N}`: k-space window centered at index `1`.

### Returns
- `v`: homodyne filtered complex (multi-echo) data
"""
function homodyne!(
    v::AbstractArray{<:Complex{<:AbstractFloat}, N},
    u::AbstractArray{<:Complex{<:AbstractFloat}, N},
    w::AbstractArray{<:AbstractFloat, M},
) where {N, M}
    M <= N || throw(DimensionMismatch("w must have at most $N dimensions, got $M"))

    checkshape(v, u, (:v, :u))
    checkshape(axes(w), axes(v)[1:M], (:w, :v))

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_fft(u, 1:M)
    iP = plan_ifft!(v, 1:M)

    mul!(v, P, u)

    if M == N
        @batch for I in eachindex(v, w)
            v[I] *= w[I]
        end

    else
        @batch for s in CartesianIndices(axes(v)[M+1:N])
            @simd for I in CartesianIndices(w)
                v[I,s] *= w[I]
            end
        end
    end

    v = iP * v
    @batch for I in eachindex(v, u)
        v[I] = u[I] / v[I]
    end

    return v
end

"""
    makewindow(
        sz::NTuple{2, Integer},
        wsz::Union{Integer, NTuple{2, Integer}},
        window::Symbol,
        args...
    ) -> Array{Float64, 2}

2d k-space window.

The window is centered at index `(1,1)`. This function is a wrapper for
DSP.jl's `makewindow(window, n = wsz, padding = sz - wsz, zerophase=true)`.

### Arguments
- `sz::NTuple{2, Integer}`: array size
- `wsz::Union{Integer, NTuple{2, Integer}}`: window size
- `window::Symbol`: window function
    - `:fermi`: `w = 1 / (1 + exp((x-radius) / width)), x ∈ [-0.5, 0.5]`
    - `:gaussian`: `w = exp(-0.5*(x/σ)^2), x ∈ [-0.5, 0.5]`
    - `:hamming`: `w = 0.54 + 0.46*cos(2π*x), x ∈ [-0.5, 0.5]`
    - `:hann`: `w = 0.5*(1 + cos(2π*x)), x ∈ [-0.5, 0.5]`
- `args...`:
    - `window == :fermi`: `radius::Real, width::Real`
    - `window == :gaussian`: `σ::Real`
    - otherwise: unused

### Returns
- `Array{Float64, 2}:` window
"""
function makewindow(
    sz::NTuple{2, Integer},
    wsz::Union{Integer, NTuple{2, Integer}},
    window::Symbol,
    args...
)
    checkopts(window, (:fermi, :gaussian, :hamming, :hann, :hanning), :window)

    n = wsz isa Integer ? (wsz, wsz) : wsz
    all(n .<= sz) || throw(ArgumentError("window size must be smaller than or equal to array size"))

    pad = sz .- n

    if window == :fermi
        w = fermi(n, args...; padding=pad, zerophase=true)

    elseif window == :gaussian
        w = Windows.gaussian(n, args...; padding=pad, zerophase=true)

    elseif window == :hann || window == :hanning
        w = Windows.hanning(n; padding=pad, zerophase=true)

    elseif window == :hamming
        w = Windows.hamming(n; padding=pad, zerophase=true)

    end

    return w
end


function fermi(
    n::Integer,
    r::Real,
    w::Real;
    padding::Integer = 0,
    zerophase::Bool = false
)
    r > 0 || throw(ArgumentError("r must be positive"))
    w > 0 || throw(ArgumentError("w must be positive"))

    iw = 1 / w
    Windows.makewindow(n, padding, zerophase) do x
        1 / (1 + exp((x-r)*iw))
    end
end

function fermi(
    n::NTuple{2, Integer},
    r::Union{Real, NTuple{2, Real}},
    w::Union{Real, NTuple{2, Real}};
    padding::Union{Integer, NTuple{2, Integer}} = 0,
    zerophase::Union{Bool, NTuple{2, Bool}} = false
)
    r2 = r isa Real ? (r, r) : r
    w2 = w isa Real ? (w, w) : w
    p2 = padding isa Integer ? (padding, padding) : padding
    z2 = zerophase isa Bool ? (zerophase, zerophase) : zerophase

    w1 = fermi(n[1], r2[1], w2[1]; padding=p2[1], zerophase=z2[1])
    w2 = fermi(n[2], r2[2], w2[2]; padding=p2[2], zerophase=z2[2])

    return w1 * w2'
end

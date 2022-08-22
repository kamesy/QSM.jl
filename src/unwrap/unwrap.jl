include("laplacian.jl")


@inline _wrap(x) = rem2pi(x, RoundNearest)

# faster, less accurate rem2pi(x, RoundNearest)
# sufficient accuracy for unwrapping MRI phase
@inline function _wrap(x::Float64)
    ax = abs(x)
    if ax < π
        x
    elseif ax < twoπ
        x - flipsign(Float64(twoπ), x)
    else
        n = round(Float64(inv2π) * x, RoundNearest)
        muladd(Float64(-twoπ), n, x)
    end
end

@inline function _wrap(x::Float32)
    Float32(_wrap(Float64(x)))
end

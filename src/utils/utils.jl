include("fd.jl")
include("fsl.jl")
include("kernels.jl")
include("lsmr.jl")
include("multi_echo.jl")
include("poisson_solver/poisson_solver.jl")
include("r2star.jl")


#####
##### FFT helpers
#####

const FAST_FFT_FACTORS = (2, 3, 5, 7)

nextfastfft(n::Real) = nextprod(FAST_FFT_FACTORS, n)
nextfastfft(ns) = nextfastfft.(ns)

"""
    fastfftsize(
        sz::NTuple{N, Integer},
        ksz::NTuple{M, Integer} = ntuple(_ -> 0, Val(N));
        rfft::Bool = false
    ) -> NTuple{N, Integer}

Next fast fft size greater than or equal to `sz` for convolution with a
kernel of size `ksz`.

### Arguments
- `x::AbstractArray{T, N}`: array to pad
- `ksz::NTuple{M, Integer} = ntuple(_ -> 0, Val(N))`: convolution kernel size
    - `ksz[n] < 0`: no padding for dimension n

### Keywords
- `rfft::Bool = false`: force first dimension to be even (`true`)

### Returns
- `NTuple{N, Integer}`: fast fft size
"""
function fastfftsize(
    sz::NTuple{N, Integer},
    ksz::NTuple{M, Integer} = ntuple(_ -> 0, Val(N));
    rfft::Bool = false
) where {N, M}
    i1 = findfirst(>(-1), ksz)

    if i1 === nothing
        return sz
    end

    szp = ntuple(Val(N)) do i
        if i > M || ksz[i] < 0
            sz[i]

        # FFTW's rfft strongly prefers even numbers in the first dimension
        elseif i == i1 && rfft
            s0 = nextfastfft(sz[i] + max(ksz[i], 1) - 1)
            s = s0
            for _ in 1:3
                iseven(s) && break
                s = nextfastfft(s+1)
            end
            s = isodd(s) ? s0 + 1 : s

        else
            nextfastfft(sz[i] + max(ksz[i], 1) - 1)
        end
    end

    return szp
end


#####
##### Padding
#####

"""
    padfastfft(
        x::AbstractArray{T, N},
        ksz::NTuple{M, Integer} = ntuple(_ -> 0, Val(N));
        pad::Symbol = :fill,
        val = zero(T),
        rfft::Bool = false,
    ) -> typeof(similar(x, szp))

Pad array `x` to a fast fft size for convolution with a kernel of size `ksz`,
keeping the array centered at `n÷2+1`.

### Arguments
- `x::AbstractArray{T, N}`: array to pad
- `ksz::NTuple{M, Integer} = ntuple(_ -> 0, Val(N))`: convolution kernel size
    - `ksz[n] < 0`: no padding for dimension n

### Keywords
- `pad::Symbol = :fill`: padding method
    - `:fill`
    - `:circular`
    - `:replicate`
    - `:symmetric`
    - `:reflect`
- `val = 0`: pads array with `val` if `pad = :fill`
- `rfft::Bool = false`: force first dimension to be even (`true`)

### Returns
- `typeof(similar(x, szp))`: padded array
"""
function padfastfft(
    x::AbstractArray{T, N},
    ksz::NTuple{M, Integer} = ntuple(_ -> 0, Val(N));
    pad::Symbol = :fill,
    val = zero(T),
    rfft::Bool = false,
) where {N, M, T}
    sz = size(x)
    szp = fastfftsize(sz, ksz, rfft=rfft)
    return sz == szp ? tcopy(x) : padarray!(similar(x, szp), x, pad, val)
end


"""
    padarray!(
        xp::AbstractArray{Txp, N},
        x::AbstractArray{Tx, N},
        pad::Symbol = :fill,
        val = 0
    ) -> xp

Pad array keeping it centered at `n÷2+1`.

### Arguments
- `xp::AbstractArray{Txp, N}`: padded array
- `x::AbstractArray{Tx, N}`: array to pad
- `pad::Symbol = :fill`: padding method
    - `:fill`
    - `:circular`
    - `:replicate`
    - `:symmetric`
    - `:reflect`
- `val = 0`: pads array with `val` if `pad = :fill`

### Returns
- `xp`: padded array
"""
function padarray!(
    xp::AbstractArray{Txp, N},
    x::AbstractArray{Tx, N},
    pad::Symbol = :fill,
    val = zero(Txp)
) where {N, Txp, Tx}
    sz = size(x)
    szp = size(xp)
    all(szp .>= sz) || throw(DimensionMismatch())

    if szp == sz
        return tcopyto!(xp, x)
    end

    valT = convert(Txp, val)
    getindex_pad =
        pad == :fill      ? (_...) -> valT :
        pad == :circular  ? getindex_circular :
        pad == :replicate ? getindex_replicate :
        pad == :symmetric ? getindex_symmetric :
        pad == :reflect   ? getindex_reflect :
        checkopts(pad, (:fill, :circular, :replicate, :symmetric, :reflect), :pad)

    return _padarray_kernel!(xp, x, getindex_pad)
end

function _padarray_kernel!(xp::AbstractArray, x::AbstractArray, getindex_pad)
    ax = axes(x)
    lo = map(first, ax)
    hi = map(last, ax)
    ΔI = CartesianIndex((size(xp) .- size(x) .+ 1) .>> 1)

    # TODO: disable threading for small xp
    @batch for Ip in CartesianIndices(xp)
        I = Ip - ΔI
        if any(map(∉, I.I, ax))
            xp[Ip] = getindex_pad(x, I, lo, hi)
        else
            xp[Ip] = x[I]
        end
    end

    return xp
end

@propagate_inbounds function getindex_circular(x, I, lo, hi)
    x[CartesianIndex(map(I.I, lo, hi) do i, l, h
        mod(i - l, h) + l
    end)]
end

@propagate_inbounds function getindex_replicate(x, I, lo, hi)
    x[CartesianIndex(map(I.I, lo, hi) do i, l, h
        clamp(i, l, h)
    end)]
end

@propagate_inbounds function getindex_symmetric(x, I, lo, hi)
    x[CartesianIndex(map(I.I, lo, hi) do i, l, h
        i < l ? 2*l - 1 - i :
        i > h ? 2*h + 1 - i : i
    end)]
end

@propagate_inbounds function getindex_reflect(x, I, lo, hi)
    x[CartesianIndex(map(I.I, lo, hi) do i, l, h
        i < l ? 2*l - i :
        i > h ? 2*h - i : i
    end)]
end


"""
    unpadarray(
        xp::AbstractArray{T, N},
        sz::NTuple{N, Integer}
    ) -> typeof(similar(xp, sz))

Extract array of size `sz` centered at `n÷2+1` from `xp`.
"""
function unpadarray(
    xp::AbstractArray{T, N},
    sz::NTuple{N, Integer}
) where {T, N}
    all(sz .<= size(xp)) || throw(DimensionMismatch())
    return unpadarray!(similar(xp, sz), xp)
end

"""
    unpadarray!(
        x::AbstractArray{Tx, N},
        xp::AbstractArray{Txp, N},
        sz::NTuple{N, Integer}
    ) -> x

Extract array centered at `n÷2+1` from `xp` into `x`.
"""
function unpadarray!(
    x::AbstractArray{Tx, N},
    xp::AbstractArray{Txp, N},
) where {N, Tx, Txp}
    sz = size(x)
    szp = size(xp)
    all(sz .<= szp) || throw(DimensionMismatch())

    ΔI = CartesianIndex((szp .- sz .+ 1) .>> 1)
    return copyto!(x, CartesianIndices(x), xp, CartesianIndices(x) .+ ΔI)
end


#####
##### Mask stuff
#####

"""
    crop_mask(
        x::AbstractArray,
        m::AbstractArray = x;
        out = 0
    ) -> typeof(x[...])

Crop array to mask.

### Arguments
- `x::AbstractArray`: array to be cropped
- `m::AbstractArray`: mask

### Keywords
- `out = 0`: value in `m` considered outside

### Returns
- `typeof(x[...])`: cropped array
"""
function crop_mask(x::AbstractArray, m::AbstractArray{T} = x; out = zero(T)) where {T}
    checkshape(x, m, (:x, :m))
    Rc = crop_indices(m, out)
    xc = tcopyto!(similar(x, size(Rc)), @view(x[Rc]))
    return xc
end

"""
    crop_indices(x::AbstractArray, out = 0) -> CartesianIndices

Indices to crop mask.

### Arguments
- `x::AbstractArray`: mask
- `out = 0`: value in `x` considered outside

### Returns
- `CartesianIndices`: indices to crop mask
"""
function crop_indices(x::AbstractArray{T, N}, out = zero(T)) where {T, N}
    outT = convert(T, out)

    cmp, pred = if T <: Bool
        identity, outT ? (!) : identity
    elseif T <: Integer
        !=(outT), identity
    else
        !≈(outT), identity
    end

    return CartesianIndices(ntuple(Val(N)) do d
        Rd = mapreduce(cmp, |, x, dims = [i for i in 1:N if i != d])
        R = Array(vec(Rd))
        findfirst(pred, R):findlast(pred, R)
    end)
end

# specialize for 3d arrays. ~30% faster
function crop_indices(x::Array{T, 3}, out = zero(T)) where {T}
    outT = convert(T, out)

    cmp, pred = if T <: Bool
        identity, outT ? (!) : identity
    elseif T <: Integer
        !=(outT), identity
    else
        !≈(outT), identity
    end

    R1 = mapreduce(cmp, |, x, dims=1, init=false)
    R2 = mapreduce(cmp, |, x, dims=2, init=false)

    Rx = mapreduce(identity, |, R2, dims=3, init=false) |> vec
    Ry = mapreduce(identity, |, R1, dims=3, init=false) |> vec
    Rz = mapreduce(identity, |, R1, dims=2, init=false) |> vec

    return CartesianIndices((
        findfirst(pred, Rx):findlast(pred, Rx),
        findfirst(pred, Ry):findlast(pred, Ry),
        findfirst(pred, Rz):findlast(pred, Rz),
    ))
end


"""
    erode_mask(mask::AbstractArray{Bool, 3}, iter::Integer = 1) -> typeof(similar(mask))

Erode binary mask using an 18-stencil cube.

### Arguments
- `mask::AbstractArray{Bool, 3}`: binary mask
- `iter::Integer = 1`: erode `iter` times

### Returns
- `typeof(similar(mask))`: eroded binary mask
"""
erode_mask(mask::AbstractArray{Bool, 3}, iter::Integer = 1) =
    erode_mask!(tzero(mask), mask, iter)

"""
    erode_mask!(
        emask::AbstractArray{Bool, 3},
        mask::AbstractArray{Bool, 3},
        iter::Integer = 1
    ) -> emask

Erode binary mask using an 18-stencil cube.

### Arguments
- `emask::AbstractArray{Bool, 3}`: eroded binary mask
- `mask::AbstractArray{Bool, 3}`: binary mask
- `iter::Integer = 1`: erode `iter` times

### Returns
- `emask`: eroded binary mask
"""
function erode_mask!(
    m1::AbstractArray{Bool, 3},
    m0::AbstractArray{Bool, 3},
    iter::Integer = 1
)
    checkshape(m1, m0, (:emask, :mask))

    if iter < 1
        return tcopyto!(m1, m0)
    end

    if iter > 1
        m0 = tcopy(m0)
    end

    nx, ny, nz = size(m0)
    for t in 1:iter
        @batch for k in 1+t:nz-t
            for j in 1+t:ny-t
                for i in 1+t:nx-t
                    m1[i,j,k] = __erode_kernel(m0, i, j, k)
                end
            end
        end

        if t < iter
            tcopyto!(m0, m1)
        end
    end

    return m1
end

@generated function __erode_kernel(m0, i, j, k)
    x = :(true)
    for _k in -1:1
        for _j in -1:1
            for _i in -1:1
                _i != 0 && _j != 0 && _k != 0 && continue
                x = :($x && m0[i+$_i, j+$_j, k+$_k])
            end
        end
    end

    quote
        Base.@_inline_meta
        return @inbounds $x
    end
end


#####
##### Misc
#####

"""
    psf2otf(
        psf::AbstractArray{<:Number, N},
        sz::NTuple{N, Integer} = size(psf);
        rfft::Bool = false,
    ) -> otf

Implementation of MATLAB's `psf2otf` function.

### Arguments
- `psf::AbstractArray{T<:Number, N}`: point-spread function
- `sz::NTuple{N, Integer}`: size of output array; must not be smaller than `psf`

### Keywords
- `rfft::Bool = false`:
    - `T<:Real`: compute `fft` (`false`) or `rfft` (`true`)
    - `T<:Complex`: unused

### Returns
- `otf`: optical transfer function
"""
function psf2otf(
    k::AbstractArray{T, N},
    sz::NTuple{N, Integer} = size(k);
    rfft::Bool = false,
) where {T<:Number, N}
    szk = size(k)
    all(szk .<= sz) || throw(DimensionMismatch())

    # zero pad
    if szk == sz
        _kp = k
    else
        _kp = tfill!(similar(k, sz), zero(T))
        @batch minbatch=1024 for I in CartesianIndices(k)
            _kp[I] = k[I]
        end
    end

    # shift so center of k is at index 1
    kp = circshift!(tzero(_kp), _kp, .-szk.÷2)

    # fft
    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = T <: Real && rfft ? plan_rfft(kp) : plan_fft(kp)

    K = P*kp

    # discard imaginary part if within roundoff error
    nops = length(k)*sum(log2, szk)
    if maximum(x -> abs(imag(x)), K) / maximum(abs2, K) ≤ nops*eps(T)
        tmap!(real, K)
    end

    return K
end


function edge_indices(
    x::AbstractArray{T, N},
    mask::Union{Nothing, AbstractArray{Bool, N}} = nothing
) where {T, N}
    edge_indices(axes(x), mask)
end

function edge_indices(
    outer::NTuple{N, AbstractUnitRange{Int}},
    mask::Union{Nothing, AbstractArray{Bool, N}} = nothing
) where {N}
    frst = map(first, outer)
    lst = map(last, outer)
    stp = map(step, outer)
    inner = ntuple(d -> frst[d]+stp[d]:lst[d]-stp[d], Val(N))
    edge_indices(outer, inner, mask)
end

function edge_indices(
    outer::NTuple{N, AbstractUnitRange{Int}},
    inner::NTuple{N, AbstractUnitRange{Int}},
    mask::Union{Nothing, AbstractArray{Bool, N}} = nothing
) where {N}
    all(first.(inner) .∈ outer) && all(last.(inner) .∈ outer) ||
        throw(DimensionMismatch("inner$inner must be in the interior of outer$outer"))

    mask !== nothing && checkshape(outer, axes(mask), (:outer, :mask))

    if mask === nothing
        n = prod(map(length, outer)) - prod(map(length, inner))

    else
        m = Ref(0)
        _edgeloop(outer, inner) do I...
            if (@inbounds mask[I...])
                m[] += 1
            end
        end
        n = m[]
    end

    if iszero(n)
        return Vector{NTuple{N, Int}}()
    end

    E = Vector{NTuple{N, Int}}(undef, n)
    i = Ref(0)
    _edgeloop(outer, inner) do I...
        if mask === nothing || (@inbounds mask[I...])
            @inbounds E[i[] += 1] = I
        end
    end

    return E
end


# totally necessary unrolled TiledIteration.EdgeIterator loop
edgeloop(f!, outer::CartesianIndices, inner::CartesianIndices) =
    edgeloop(f!, outer.indices, inner.indices)

@inline function edgeloop(
    f!,
    outer::NTuple{N, AbstractUnitRange{Int}},
    inner::NTuple{N, AbstractUnitRange{Int}},
) where {N}
    all(first.(inner) .∈ outer) && all(last.(inner) .∈ outer) ||
        throw(DimensionMismatch("inner$inner must be in the interior of outer$outer"))
    _edgeloop(f!, outer, inner)
end

@generated function _edgeloop(
    f!::F,
    outer::NTuple{N, AbstractUnitRange{Int}},
    inner::NTuple{N, AbstractUnitRange{Int}},
) where {F, N}
    N == 0 && return :(nothing)
    I = ntuple(d -> Symbol(:I, d), Val(N))

    exf! = quote
        f!($(I...))
    end

    ex = quote
        for $(I[1]) in first(outer[1]):first(inner[1])-1
            $exf!
        end
        for $(I[1]) in last(inner[1])+1:last(outer[1])
            $exf!
        end
    end

    for d in 2:N
        expp = exf!
        for n in 1:d-1
            expp = quote
                for $(I[n]) in outer[$n]
                    $expp
                end
            end
        end

        ex = quote
            for $(I[d]) in first(outer[$d]):first(inner[$d])-1
                $expp
            end
            for $(I[d]) in inner[$d]
                $ex
            end
            for $(I[d]) in last(inner[$d])+1:last(outer[$d])
                $expp
            end
        end
    end

    quote
        Base.@_inline_meta
        $ex
    end
end


macro bfor(ex)
    _bfor(ex)
end

function _bfor(ex)
    valid = true
    vars = Symbol[]
    ivars = Symbol[]

    postwalk(ex) do x
        if valid && @capture(x, var_[Is__])
            valid = length(Is) == 1
            push!(vars, var)
            push!(ivars, Is...)
        end
        return x
    end

    vars = unique!(sort!(vars))
    ivars = unique!(sort!(ivars))

    if isempty(ivars)
        throw(ArgumentError("loop index not found"))
    end

    if !valid || length(ivars) > 1
        throw(ArgumentError("multiple indices not supported"))
    end

    loop = :(
        for $(first(ivars)) in eachindex($(vars...))
            $ex
        end
    )

    Polyester.enclose(loop, 0, 1, :core, (Symbol(""), :Any), Polyester)
end


#####
##### Multi-threaded Base utilities
#####

tzero(x) = zero(x)
tzero(x::AbstractArray{T}) where {T} = tfill!(similar(x, typeof(zero(T))), zero(T))


tfill!(A, x) = fill!(A, x)

function tfill!(A::AbstractArray{T}, x) where {T}
    xT = convert(T, x)
    @batch minbatch=1024 for I in eachindex(A)
        @inbounds A[I] = xT
    end
    return A
end


tmap(f, iters...) = map(f, iters...)

function tmap(f, A::AbstractArray)
    isempty(A) && return similar(A, 0)
    dest = similar(A, typeof(f(A[1])))
    return tmap!(f, dest, A)
end


tmap!(f, dest, iters...) = map!(f, dest, iters...)
tmap!(f, A::AbstractArray) = tmap!(f, A, A)

function tmap!(f::F, dest::AbstractArray, A::AbstractArray) where {F}
    checkshape(Bool, dest, A) || return map!(f, dest, A)
    @batch minbatch=1024 for I in eachindex(dest, A)
        val = f(@inbounds A[I])
        @inbounds dest[I] = val
    end
    return dest
end


tcopy(x) = copy(x)
tcopy(x::AbstractArray) = tcopyto!(similar(x), x)


tcopyto!(dest, src) = copyto!(dest, src)

function tcopyto!(dest::AbstractArray, src::AbstractArray)
    checkshape(Bool, dest, src) || return copyto!(dest, src)
    @batch minbatch=1024 for I in eachindex(dest, src)
        @inbounds dest[I] = src[I]
    end
    return dest
end


#####
##### Error checking
#####

checkshape(
    ::Type{Bool},
    ::Tuple{},
    ::Tuple{}
) = true

checkshape(
    ::Tuple{},
    ::Tuple{},
    ::NTuple{2, Union{Symbol, AbstractString}} = (:a, :b)
) = nothing


function checkshape(
    ::Type{Bool},
    a::NTuple{Na, Integer},
    b::NTuple{Nb, Integer},
) where {Na, Nb}
    Na < Nb && return checkshape(Bool, b, a)
    return all(i -> a[i] == b[i], 1:Nb) && all(i -> a[i] == 1, Nb+1:Na)
end

function checkshape(
    a::NTuple{Na, Integer},
    b::NTuple{Nb, Integer},
    vars::NTuple{2, Union{Symbol, AbstractString}} = (:a, :b),
) where {Na, Nb}
    if !checkshape(Bool, a, b)
        na, nb = vars
        throw(DimensionMismatch("shape must match: $na has dims $a, $nb has dims $b"))
    end
    return nothing
end


function checkshape(::Type{Bool}, a::AbstractArray, b::AbstractArray)
    checkshape(Bool, axes(a), axes(b))
end

function checkshape(
    a::AbstractArray,
    b::AbstractArray,
    vars::NTuple{2, Union{Symbol, AbstractString}} = (:a, :b)
)
    checkshape(axes(a), axes(b), vars)
end

function checkshape(
    ::Type{Bool},
    a::NTuple{Na, AbstractUnitRange},
    b::NTuple{Nb, AbstractUnitRange},
) where {Na, Nb}
    Na < Nb && return checkshape(Bool, b, a)
    return all(i -> a[i] == b[i], 1:Nb) && all(i -> a[i] == 1:1, Nb+1:Na)
end

function checkshape(
    a::NTuple{Na, AbstractUnitRange},
    b::NTuple{Nb, AbstractUnitRange},
    vars::NTuple{2, Union{Symbol, AbstractString}} = (:a, :b),
) where {Na, Nb}
    if !checkshape(Bool, a, b)
        na, nb = vars
        throw(DimensionMismatch("shape must match: $na has dims $a, $nb has dims $b"))
    end
    return nothing
end


checkopts(::Type{Bool}, o::T, opts::NTuple{N, T}) where {N, T} = o ∈ opts

function checkopts(
    o::T,
    opts::NTuple{N, T},
    var::Union{Symbol, AbstractString} = :x
) where {N, T}
    if !checkopts(Bool, o, opts)
        pp = map((o, opts...)) do x
            x isa Symbol ? ":$x" :
            x isa AbstractString ? "\"$x\"" : "$x"
        end
        s1 = first(pp)
        s2 = join(Base.tail(pp), ", ", " ")
        throw(ArgumentError("$var must be one of $s2, got $s1"))
    end
    return nothing
end

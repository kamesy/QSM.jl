struct Poisson2{T<:AbstractFloat, N, TD<:AbstractArray{Bool, N}, TR8, TR27, TRI}
    dx::NTuple{N, T}
    idx2::NTuple{N, T}
    D::T
    iD::T
    interior::TD
    boundary::TD
    R8::TR8
    R27::TR27
    RI::TRI
end

Poisson2(interior, dx) = Poisson2(Float64, interior, dx)

function Poisson2(
    ::Type{T},
    interior::AbstractArray{Bool, N},
    dx::NTuple{N, Real}
) where {T<:AbstractFloat, N}
    dxT = convert.(T, dx)

    idx2 = inv.(dxT.*dxT)
    D = -2*sum(idx2)

    boundary = boundary_mask(interior)

    ax = map(a -> first(a)+1:last(a)-1, axes(interior))

    tsz = padded_tilesize(T, ntuple(_ -> 2, Val(N)), 1)
    R8 = vec(collect(TileIterator(ax, tsz)))

    tsz = padded_tilesize(T, ntuple(_ -> 3, Val(N)), 1)
    R27 = vec(collect(TileIterator(ax, tsz)))

    i = 0
    RI = Vector{eltype(eachindex(interior))}(undef, sum(interior))
    @inbounds for I in eachindex(interior)
        if interior[I]
            RI[i += 1] = I
        end
    end

    return Poisson2(dxT, idx2, D, inv(D), interior, boundary, R8, R27, RI)
end


Base.size(A::Poisson2) = size(A.interior)
Base.eltype(::Poisson2{T}) where {T} = T


function Base.:(*)(
    A::Poisson2{<:AbstractFloat, 3},
    x::AbstractArray{<:AbstractFloat, 3}
)
    return mul!(tzero(x), A, x)
end

function Base.:(*)(
    A::Poisson2{<:AbstractFloat, 3},
    x::AbstractArray{<:AbstractFloat, 4}
)
    return mul!(tzero(x), A, x)
end

function LinearAlgebra.mul!(
    d2x::AbstractArray{<:AbstractFloat, 3},
    A::Poisson2{<:AbstractFloat, 3},
    x::AbstractArray{T, 3},
) where {T<:AbstractFloat}
    D = convert(T, A.D)
    idx2, idy2, idz2 = convert.(T, A.idx2)

    # @threads ~10% faster here
    # Polyester needs A.interior inside or the loop slows down by 100%???
    # ie. slow down when m = A.interior ... if m[i,j,k] ...
    @batch for (I, J, K) in A.R8
        for k in K
            for j in J
                for i in I
                    z1 = x[i,j,k-1]
                    y1 = x[i,j-1,k]
                    x1 = x[i-1,j,k]
                    x0 = x[i,j,k]
                    x2 = x[i+1,j,k]
                    y2 = x[i,j+1,k]
                    z2 = x[i,j,k+1]

                    if A.interior[i,j,k]
                        Δz = z1 + z2
                        Δy = y1 + y2
                        Δx = x1 + x2

                        Δ = D * x0
                        Δ = muladd(idz2, Δz, Δ)
                        Δ = muladd(idy2, Δy, Δ)
                        Δ = muladd(idx2, Δx, Δ)

                        d2x[i,j,k] = Δ
                    end
                end
            end
        end
    end

    return d2x
end

function LinearAlgebra.mul!(
    d2x::AbstractArray{<:AbstractFloat, 4},
    A::Poisson2{<:AbstractFloat, 3},
    x::AbstractArray{T, 4},
) where {T<:AbstractFloat}
    D = convert(T, A.D)
    idx2, idy2, idz2 = convert.(T, A.idx2)

    # @threads ~10% faster here
    # Polyester needs A.interior inside or the loop slows down by 100%???
    # ie. slow down when m = A.interior ... if m[i,j,k] ...
    for t in axes(x, 4)
        xt = @view(x[:,:,:,t])
        d2xt = @view(d2x[:,:,:,t])
        @batch for (I, J, K) in A.R8
            for k in K
                for j in J
                    for i in I
                        z1 = xt[i,j,k-1]
                        y1 = xt[i,j-1,k]
                        x1 = xt[i-1,j,k]
                        x0 = xt[i,j,k]
                        x2 = xt[i+1,j,k]
                        y2 = xt[i,j+1,k]
                        z2 = xt[i,j,k+1]

                        if A.interior[i,j,k]
                            Δz = z1 + z2
                            Δy = y1 + y2
                            Δx = x1 + x2

                            Δ = D * x0
                            Δ = muladd(idz2, Δz, Δ)
                            Δ = muladd(idy2, Δy, Δ)
                            Δ = muladd(idx2, Δx, Δ)

                            d2xt[i,j,k] = Δ
                        end
                    end
                end
            end
        end
    end

    return d2x
end


function residual!(
    r::AbstractArray{<:AbstractFloat, 3},
    b::AbstractArray{<:AbstractFloat, 3},
    A::Poisson2{<:AbstractFloat, 3},
    x::AbstractArray{T, 3},
) where {T<:AbstractFloat}
    D = convert(T, -A.D)
    idx2 = convert(T, -A.idx2[1])
    idy2 = convert(T, -A.idx2[2])
    idz2 = convert(T, -A.idx2[3])

    # @threads ~10% faster here
    # Polyester needs A.interior inside or the loop slows down by 100%???
    # ie. slow down when m = A.interior ... if m[i,j,k] ...
    @batch for (I, J, K) in A.R8
        for k in K
            for j in J
                for i in I
                    z1 = x[i,j,k-1]
                    y1 = x[i,j-1,k]
                    x1 = x[i-1,j,k]
                    x0 = x[i,j,k]
                    x2 = x[i+1,j,k]
                    y2 = x[i,j+1,k]
                    z2 = x[i,j,k+1]

                    if A.interior[i,j,k]
                        Δz = z1 + z2
                        Δy = y1 + y2
                        Δx = x1 + x2

                        Δ = muladd(D, x0, b[i,j,k])
                        Δ = muladd(idz2, Δz, Δ)
                        Δ = muladd(idy2, Δy, Δ)
                        Δ = muladd(idx2, Δx, Δ)

                        r[i,j,k] = Δ
                    end
                end
            end
        end
    end

    return r
end


function __norm_residual(
    b::AbstractArray{<:AbstractFloat, 3},
    A::Poisson2{<:AbstractFloat, 3},
    x::AbstractArray{T, 3},
) where {T<:AbstractFloat}
    m = A.interior

    D = convert(T, -A.D)
    idx2 = convert(T, -A.idx2[1])
    idy2 = convert(T, -A.idx2[2])
    idz2 = convert(T, -A.idx2[3])

    @batch threadlocal=zero(T)::T for (I, J, K) in A.R8
        for k in K
            for j in J
                for i in I
                    z1 = x[i,j,k-1]
                    y1 = x[i,j-1,k]
                    x1 = x[i-1,j,k]
                    x0 = x[i,j,k]
                    x2 = x[i+1,j,k]
                    y2 = x[i,j+1,k]
                    z2 = x[i,j,k+1]

                    if m[i,j,k]
                        Δz = z1 + z2
                        Δy = y1 + y2
                        Δx = x1 + x2

                        Δ = muladd(D, x0, b[i,j,k])
                        Δ = muladd(idz2, Δz, Δ)
                        Δ = muladd(idy2, Δy, Δ)
                        Δ = muladd(idx2, Δx, Δ)

                        threadlocal = muladd(Δ, Δ, threadlocal)
                    end
                end
            end
        end
    end

    return sqrt(sum(threadlocal::Vector{T}))
end


boundary_mask(m::AbstractArray{Bool, 3}) = boundary_mask!(tzero(m), m)

function boundary_mask!(
    mb::AbstractArray{Bool, 3},
    m::AbstractArray{Bool, 3}
)
    sz = size(m)
    nx, ny, nz = sz
    nxc, nyc, nzc = restrict_size(sz)

    b = tzero(mb)

    outer = ntuple(n -> 2:sz[n]-1, Val(3))
    inner = ntuple(n -> 3:sz[n]-2-Int(iseven(sz[n])), Val(3))
    _edgeloop(outer, inner) do i, j, k
        @inbounds b[i,j,k] = m[i,j,k]
    end

    # interior
    @batch minbatch=8 for kc in 2:nzc-1
        k = (kc << 1) - 1
        for jc in 2:nyc-1
            j = (jc << 1) - 1
            for ic in 2:nxc-1
                i = (ic << 1) - 1

                if !m[i,j,k]     ||
                   !m[i+1,j,k]   ||
                   !m[i,j+1,k]   ||
                   !m[i+1,j+1,k] ||
                   !m[i,j,k+1]   ||
                   !m[i+1,j,k+1] ||
                   !m[i,j+1,k+1] ||
                   !m[i+1,j+1,k+1]

                    b[i,j,k]       = m[i,j,k]
                    b[i+1,j,k]     = m[i+1,j,k]
                    b[i,j+1,k]     = m[i,j+1,k]
                    b[i+1,j+1,k]   = m[i+1,j+1,k]
                    b[i,j,k+1]     = m[i,j,k+1]
                    b[i+1,j,k+1]   = m[i+1,j,k+1]
                    b[i,j+1,k+1]   = m[i,j+1,k+1]
                    b[i+1,j+1,k+1] = m[i+1,j+1,k+1]
                end
            end
        end
    end

    @batch minbatch=8 for k in 2:nz-1
        for j in 2:ny-1
            for i in 2:nx-1
                if m[i,j,k]
                    mb[i,j,k] =
                        b[i,j,k-1] ||
                        b[i,j-1,k] ||
                        b[i-1,j,k] ||
                        b[i,j,k] ||
                        b[i+1,j,k] ||
                        b[i,j+1,k] ||
                        b[i,j,k+1]
                end
            end
        end
    end

    return mb
end


function tfill!(x::AbstractArray{T, N}, v, A::Poisson2{<:AbstractFloat, N}) where {T, N}
    vT = convert(T, v)
    @batch minbatch=1024 for I in A.RI
        x[I] = vT
    end
    return x
end

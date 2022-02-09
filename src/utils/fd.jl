#####
##### Gradient, forward finite-differences, periodic boundaries
#####

"""
    gradfp(
        u::AbstractArray{<:AbstractFloat, 3},
        h::NTuple{3, Real}
    ) -> NTuple{3, typeof(similar(u))}

First order forward difference gradient with periodic boundaries.

### Arguments
- `u::AbstractArray{<:AbstractFloat, 3}`: input array
- `h::NTuple{3, Real}`: grid spacing

### Returns
- `dx::typeof(similar(u))`: x-component of gradient of `u`
- `dy::typeof(similar(u))`: y-component of gradient of `u`
- `dz::typeof(similar(u))`: z-component of gradient of `u`
"""
gradfp(u::AbstractArray{<:AbstractFloat, 3}, h::NTuple{3, Real}) =
    gradfp!(similar(u), similar(u), similar(u), u, h)

"""
    gradfp!(
        dx::AbstractArray{<:AbstractFloat, 3},
        dy::AbstractArray{<:AbstractFloat, 3},
        dz::AbstractArray{<:AbstractFloat, 3},
        u::AbstractArray{<:AbstractFloat, 3},
        h::NTuple{3, Real},
    ) -> (dx, dy, dz)

First order forward difference gradient with periodic boundaries.

### Arguments
- `dx::AbstractArray{<:AbstractFloat, 3}`: x-component of gradient of `u`
- `dy::AbstractArray{<:AbstractFloat, 3}`: y-component of gradient of `u`
- `dz::AbstractArray{<:AbstractFloat, 3}`: z-component of gradient of `u`
- `u::AbstractArray{<:AbstractFloat, 3}`: input array
- `h::NTuple{3, Real}`: grid spacing

### Returns
- `dx`: x-component of gradient of `u`
- `dy`: y-component of gradient of `u`
- `dz`: z-component of gradient of `u`
"""
function gradfp!(dx, dy, dz, u, h)
    size(dx) == size(u) || throw(DimensionMismatch())
    size(dy) == size(u) || throw(DimensionMismatch())
    size(dz) == size(u) || throw(DimensionMismatch())
    return _gradfp!(dx, dy, dz, u, h)
end

function _gradfp!(
    dx::AbstractArray{Tdx, 3},
    dy::AbstractArray{Tdy, 3},
    dz::AbstractArray{Tdz, 3},
    u::AbstractArray{<:AbstractFloat, 3},
    h::NTuple{3, Real},
) where {Tdx<:AbstractFloat, Tdy<:AbstractFloat, Tdz<:AbstractFloat}
    nx, ny, nz = size(u)
    hx = convert(Tdx, inv(h[1]))
    hy = convert(Tdy, inv(h[2]))
    hz = convert(Tdz, inv(h[3]))

    @inbounds @batch for k in 1:nz-1
        for j in 1:ny-1
            for i in 1:nx-1
                dx[i,j,k] = hx*(u[i+1,j,k] - u[i,j,k])
                dy[i,j,k] = hy*(u[i,j+1,k] - u[i,j,k])
                dz[i,j,k] = hz*(u[i,j,k+1] - u[i,j,k])
            end

            dx[nx,j,k] = hx*(u[1,j,k] - u[nx,j,k])
            dy[nx,j,k] = hy*(u[nx,j+1,k] - u[nx,j,k])
            dz[nx,j,k] = hz*(u[nx,j,k+1] - u[nx,j,k])
        end

        for i in 1:nx-1
            dy[i,ny,k] = hy*(u[i,1,k] - u[i,ny,k])
            dx[i,ny,k] = hx*(u[i+1,ny,k] - u[i,ny,k])
            dz[i,ny,k] = hz*(u[i,ny,k+1] - u[i,ny,k])
        end

        dy[nx,ny,k] = hy*(u[nx,1,k] - u[nx,ny,k])
        dx[nx,ny,k] = hx*(u[1,ny,k] - u[nx,ny,k])
        dz[nx,ny,k] = hz*(u[nx,ny,k+1] - u[nx,ny,k])
    end

    @inbounds @batch for j in 1:ny-1
        for i in 1:nx-1
            dz[i,j,nz] = hz*(u[i,j,1] - u[i,j,nz])
            dx[i,j,nz] = hx*(u[i+1,j,nz] - u[i,j,nz])
            dy[i,j,nz] = hy*(u[i,j+1,nz] - u[i,j,nz])
        end

        dz[nx,j,nz] = hz*(u[nx,j,1] - u[nx,j,nz])
        dx[nx,j,nz] = hx*(u[1,j,nz] - u[nx,j,nz])
        dy[nx,j,nz] = hy*(u[nx,j+1,nz] - u[nx,j,nz])
    end

    @inbounds @batch for i in 1:nx-1
        dz[i,ny,nz] = hz*(u[i,ny,1] - u[i,ny,nz])
        dy[i,ny,nz] = hy*(u[i,1,nz] - u[i,ny,nz])
        dx[i,ny,nz] = hx*(u[i+1,ny,nz] - u[i,ny,nz])
    end

    @inbounds begin
        dz[nx,ny,nz] = hz*(u[nx,ny,1] - u[nx,ny,nz])
        dy[nx,ny,nz] = hy*(u[nx,1,nz] - u[nx,ny,nz])
        dx[nx,ny,nz] = hx*(u[1,ny,nz] - u[nx,ny,nz])
    end

    return dx, dy, dz
end


"""
    gradfp_adj(
        dx::AbstractArray{<:AbstractFloat, 3},
        dy::AbstractArray{<:AbstractFloat, 3},
        dz::AbstractArray{<:AbstractFloat, 3},
        h::NTuple{3, Real}
    ) -> typeof(similar(dx, promote_eltype(dx, dy, dz)))

Adjoint of first order forward difference gradient with periodic boundaries.

### Arguments
- `dx::AbstractArray{<:AbstractFloat, 3}`: x-component
- `dy::AbstractArray{<:AbstractFloat, 3}`: y-component
- `dz::AbstractArray{<:AbstractFloat, 3}`: z-component
- `h::NTuple{3, Real}`: grid spacing

### Returns
- `u::typeof(similar(dx, promote_eltype(dx, dy ,dz)))`: divergence of [dx, dy, dz]
"""
function gradfp_adj(
    dx::AbstractArray{Tdx, 3},
    dy::AbstractArray{Tdy, 3},
    dz::AbstractArray{Tdz, 3},
    h::NTuple{3, Real}
) where {Tdx, Tdy, Tdz}
    Td2u = foldl(promote_type, (Tdy, Tdz), init=Tdx)
    d2u = similar(dx, Td2u)
    gradfp_adj!(d2u, dx, dy, dz, h)
end

"""
    gradfp_adj!(
        u::AbstractArray{<:AbstractFloat, 3}
        dx::AbstractArray{<:AbstractFloat, 3},
        dy::AbstractArray{<:AbstractFloat, 3},
        dz::AbstractArray{<:AbstractFloat, 3},
        h::NTuple{3, Real}
    ) -> u

Adjoint of first order forward difference gradient with periodic boundaries.

### Arguments
- `u::AbstractArray{<:AbstractFloat, 3}`: divergence of [dx, dy, dz]
- `dx::AbstractArray{<:AbstractFloat, 3}`: x-component
- `dy::AbstractArray{<:AbstractFloat, 3}`: y-component
- `dz::AbstractArray{<:AbstractFloat, 3}`: z-component
- `h::NTuple{3, Real}`: grid spacing

### Returns
- `u`: divergence of [dx, dy, dz]
"""
function gradfp_adj!(d2u, dx, dy, dz, h)
    size(d2u) == size(dx) || throw(DimensionMismatch())
    size(d2u) == size(dy) || throw(DimensionMismatch())
    size(d2u) == size(dz) || throw(DimensionMismatch())
    return _gradfp_adj!(d2u, dx, dy, dz, h)
end

function _gradfp_adj!(
    d2u::AbstractArray{<:AbstractFloat, 3},
    dx::AbstractArray{Tdx, 3},
    dy::AbstractArray{Tdy, 3},
    dz::AbstractArray{Tdz, 3},
    h::NTuple{3, Real},
) where {Tdx, Tdy, Tdz}
    nx, ny, nz = size(d2u)
    hx = convert(Tdx, -inv(h[1]))
    hy = convert(Tdy, -inv(h[2]))
    hz = convert(Tdz, -inv(h[3]))

    @inbounds begin
        d2u[1,1,1] =
            hx*(dx[1,1,1] - dx[nx,1,1]) +
            hy*(dy[1,1,1] - dy[1,ny,1]) +
            hz*(dz[1,1,1] - dz[1,1,nz])
    end

    @inbounds @batch for i in 2:nx
        ux = dx[i,1,1] - dx[i-1,1,1]
        uy = dy[i,1,1] - dy[i,ny,1]
        uz = dz[i,1,1] - dz[i,1,nz]

        u = hx*ux
        u = muladd(hy, uy, u)
        u = muladd(hz, uz, u)
        d2u[i,1,1] = u
    end

    @inbounds @batch for j in 2:ny
        uy = dy[1,j,1] - dy[1,j-1,1]
        ux = dx[1,j,1] - dx[nx,j,1]
        uz = dz[1,j,1] - dz[1,j,nz]

        u = hy*uy
        u = muladd(hx, ux, u)
        u = muladd(hz, uz, u)
        d2u[1,j,1] = u

        for i in 2:nx
            uy = dy[i,j,1] - dy[i,j-1,1]
            ux = dx[i,j,1] - dx[i-1,j,1]
            uz = dz[i,j,1] - dz[i,j,nz]

            u = hy*uy
            u = muladd(hx, ux, u)
            u = muladd(hz, uz, u)
            d2u[i,j,1] = u
        end
    end

    @inbounds @batch for k in 2:nz
        uz = dz[1,1,k] - dz[1,1,k-1]
        ux = dx[1,1,k] - dx[nx,1,k]
        uy = dy[1,1,k] - dy[1,ny,k]

        u = hz*uz
        u = muladd(hx, ux, u)
        u = muladd(hy, uy, u)
        d2u[1,1,k] = u

        for i in 2:nx
            uz = dz[i,1,k] - dz[i,1,k-1]
            uy = dy[i,1,k] - dy[i,ny,k]
            ux = dx[i,1,k] - dx[i-1,1,k]

            u = hz*uz
            u = muladd(hy, uy, u)
            u = muladd(hx, ux, u)
            d2u[i,1,k] = u
        end

        for j in 2:ny
            uz = dz[1,j,k] - dz[1,j,k-1]
            uy = dy[1,j,k] - dy[1,j-1,k]
            ux = dx[1,j,k] - dx[nx,j,k]

            u = hz*uz
            u = muladd(hy, uy, u)
            u = muladd(hx, ux, u)
            d2u[1,j,k] = u

            for i in 2:nx
                uz = dz[i,j,k] - dz[i,j,k-1]
                uy = dy[i,j,k] - dy[i,j-1,k]
                ux = dx[i,j,k] - dx[i-1,j,k]

                u = hz*uz
                u = muladd(hy, uy, u)
                u = muladd(hx, ux, u)
                d2u[i,j,k] = u
            end
        end
    end

    return d2u
end


#####
##### Laplacian, central finite-differences
#####

"""
    lap(
        u::AbstractArray{<:AbstractFloat, 3},
        h::NTuple{3, Real}
    ) -> typeof(similar(u))

Second order central difference Laplacian.

### Arguments
- `u::AbstractArray{<:AbstractFloat, 3}`: input array
- `h::NTuple{3, Real}`: grid spacing

### Returns
- `d2u::typeof(similar(u))`: discrete Laplacian of `u`
"""
lap(u::AbstractArray{<:AbstractFloat}, h::NTuple{3, Real}) =
    lap!(tzero(u), u, h)

"""
    lap!(
        d2u::AbstractArray{<:AbstractFloat, 3},
        u::AbstractArray{<:AbstractFloat, 3},
        h::NTuple{3, Real}
    ) -> d2u

Second order central difference Laplacian.

### Arguments
- `d2u::AbstractArray{<:AbstractFloat, 3}`: discrete Laplacian of `u`
- `u::AbstractArray{<:AbstractFloat, 3}`: input array
- `h::NTuple{3, Real}`: grid spacing

### Returns
- `d2u`: discrete Laplacian of `u`
"""
function lap!(d2u, u, h)
    size(d2u) == size(u) || throw(DimensionMismatch())
    return _lap!(d2u, u, h)
end

function _lap!(
    d2u::AbstractArray{<:AbstractFloat, 3},
    u::AbstractArray{T, 3},
    h::NTuple{3, Real},
) where {T<:AbstractFloat}
    idx2 = convert(T, inv(h[1]*h[1]))
    idy2 = convert(T, inv(h[2]*h[2]))
    idz2 = convert(T, inv(h[3]*h[3]))
    D = -2*(idx2 + idy2 + idz2)

    ax = map(a -> first(a)+1:last(a)-1, axes(u))

    tsz = padded_tilesize(T, (2, 2, 2), 1)
    R = vec(collect(TileIterator(ax, tsz)))

    @inbounds @batch for (I, J, K) in R
        for k in K
            for j in J
                for i in I
                    z1 = u[i,j,k-1]
                    y1 = u[i,j-1,k]
                    x1 = u[i-1,j,k]
                    u0 = u[i,j,k]
                    x2 = u[i+1,j,k]
                    y2 = u[i,j+1,k]
                    z2 = u[i,j,k+1]

                    Δz = z1 + z2
                    Δy = y1 + y2
                    Δx = x1 + x2

                    Δ = D * u0
                    Δ = muladd(idz2, Δz, Δ)
                    Δ = muladd(idy2, Δy, Δ)
                    Δ = muladd(idx2, Δx, Δ)

                    d2u[i,j,k] = Δ
                end
            end
        end
    end

    return d2u
end

function _lap!(
    d2u::AbstractArray{<:AbstractFloat, 4},
    u::AbstractArray{T, 4},
    h::NTuple{3, Real},
) where {T<:AbstractFloat}
    idx2 = convert(T, inv(h[1]*h[1]))
    idy2 = convert(T, inv(h[2]*h[2]))
    idz2 = convert(T, inv(h[3]*h[3]))
    D = -2*(idx2 + idy2 + idz2)

    ax = map(a -> first(a)+1:last(a)-1, axes(u)[1:3])

    tsz = padded_tilesize(T, (2, 2, 2), 1)
    R = vec(collect(TileIterator(ax, tsz)))

    @inbounds for t in axes(u, 4)
        ut = @view(u[:,:,:,t])
        d2ut = @view(d2u[:,:,:,t])
        @batch for (I, J, K) in R
            for k in K
                for j in J
                    for i in I
                        z1 = ut[i,j,k-1]
                        y1 = ut[i,j-1,k]
                        x1 = ut[i-1,j,k]
                        u0 = ut[i,j,k]
                        x2 = ut[i+1,j,k]
                        y2 = ut[i,j+1,k]
                        z2 = ut[i,j,k+1]

                        Δz = z1 + z2
                        Δy = y1 + y2
                        Δx = x1 + x2

                        Δ = D * u0
                        Δ = muladd(idz2, Δz, Δ)
                        Δ = muladd(idy2, Δy, Δ)
                        Δ = muladd(idx2, Δx, Δ)

                        d2ut[i,j,k] = Δ
                    end
                end
            end
        end
    end

    return d2u
end

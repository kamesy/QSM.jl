include("multigrid/utils.jl")
include("multigrid/poisson.jl")
include("multigrid/transfer.jl")
include("multigrid/smoothers.jl")
include("multigrid/multigrid.jl")
include("mgpcg.jl")


function solve_poisson_mgpcg(
    d2u::AbstractArray{<:AbstractFloat, N},
    mask::AbstractArray{Bool, 3},
    dx::NTuple{3, Real};
    kwargs...
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return solve_poisson_mgpcg!(tzero(d2u), d2u, mask, dx; kwargs...)
end

function solve_poisson_mgpcg!(
    u::AbstractArray{<:AbstractFloat, N},
    d2u::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    dx::NTuple{3, Real};
    kwargs...
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    checkshape(u, d2u, (:u, :d2u))
    checkshape(axes(mask), axes(u)[1:3], (:mask, :u))

    # crop to avoid unnecessary work. u[!mask] = 0
    sz = size(mask)
    Rc = crop_indices(mask)

    # pad for finite diffy stencil
    I1, I2 = first(Rc), last(Rc)
    Rc = CartesianIndices(ntuple(ndims(Rc)) do n
        max(1, I1[n]-1) : min(sz[n], I2[n]+1)
    end)

    # TODO: performance test proper cutoff
    szc = size(Rc)
    _crop = maximum(sz .- szc) > 10

    if !_crop
        mc = mask
        d2uc = d2u
        uc = u
    else
        mc = similar(mask, szc)
        mc = tcopyto!(mc, @view(mask[Rc]))

        if N == 3
            d2uc = @view(d2u[Rc])
            uc = similar(u, szc)
            uc = tcopyto!(uc, @view(u[Rc]))
        else
            d2uc = @view(d2u[Rc,:])
            uc = similar(u, (szc..., size(u, 4)))
            uc = tcopyto!(uc, @view(u[Rc,:]))
        end
    end

    # solve
    A = Poisson2(T, mc, dx)
    uc = mgpcg!(uc, A, d2uc; kwargs...)

    if _crop
        if N == 3
            tcopyto!(@view(u[Rc]), uc)
        else
            tcopyto!(@view(u[Rc,:]), uc)
        end
    end

    return u
end


function solve_poisson_dct(
    d2u::AbstractArray{<:AbstractFloat, N},
    dx::NTuple{3, Real}
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return solve_poisson_dct!(similar(d2u), d2u, dx)
end

function solve_poisson_dct!(
    u::AbstractArray{<:AbstractFloat, N},
    d2u::AbstractArray{<:AbstractFloat, N},
    dx::NTuple{3, Real}
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    checkshape(u, d2u, (:u, :d2u))

    nx, ny, nz = size(d2u)[1:3]
    idx2 = inv(dx[1].*dx[1])
    idy2 = inv(dx[2].*dx[2])
    idz2 = inv(dx[3].*dx[3])

    # extreme slowdown for certain sizes with lots of threads
    # even worse for in-place, ie dct!
    FFTW.set_num_threads(max(1, FFTW_NTHREADS[]÷2))
    P = plan_dct(u, 1:3)
    iP = inv(P)

    d2û = P*d2u

    X = [2*(cospi(i)-1)*idx2 for i in range(0, step=1/nx, length=nx)]
    Y = [2*(cospi(j)-1)*idy2 for j in range(0, step=1/ny, length=ny)]
    Z = [2*(cospi(k)-1)*idz2 for k in range(0, step=1/nz, length=nz)]

    @inbounds for t in axes(d2û, 4)
        d2ût = @view(d2û[:,:,:,t])
        @batch for k in 1:nz
            for j in 1:ny
                for i in 1:nx
                    d2ût[i,j,k] *= inv(X[i] + Y[j] + Z[k])
                end
            end
        end
        # X = Y = Z = 0. Solution is not unique. Set mean to 0
        d2ût[1,1,1] = 0
    end

    # inverse dct
    u = mul!(u, iP, d2û)

    return u
end


function solve_poisson_fft(
    d2u::AbstractArray{<:AbstractFloat, N},
    dx::NTuple{3, Real}
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return solve_poisson_fft!(similar(d2u), d2u, dx)
end

function solve_poisson_fft!(
    u::AbstractArray{<:AbstractFloat, N},
    d2u::AbstractArray{<:AbstractFloat, N},
    dx::NTuple{3, Real}
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    checkshape(u, d2u, (:u, :d2u))

    nx, ny, nz = size(d2u)[1:3]
    idx2 = inv(dx[1].*dx[1])
    idy2 = inv(dx[2].*dx[2])
    idz2 = inv(dx[3].*dx[3])

    _rfft = iseven(nx)
    FFTW.set_num_threads(FFTW_NTHREADS[])

    # FFTW's rfft is extremely slow with some odd lengths in the first dim
    if _rfft
        nx = nx>>1 + 1
        P = plan_rfft(d2u, 1:3)
        iP = inv(P)
        d2û = P*d2u
    else
        d2û = tcopyto!(similar(d2u, complex(eltype(d2u))), d2u)
        P = plan_fft!(d2û, 1:3)
        iP = inv(P)
        d2û = P*d2û
    end

    X = [2*(cospi(i)-1)*idx2 for i in fftfreq(size(u, 1), 2)]
    Y = [2*(cospi(j)-1)*idy2 for j in fftfreq(size(u, 2), 2)]
    Z = [2*(cospi(k)-1)*idz2 for k in fftfreq(size(u, 3), 2)]

    @inbounds for t in axes(d2û, 4)
        d2ût = @view(d2û[:,:,:,t])
        @batch for k in 1:nz
            for j in 1:ny
                for i in 1:nx
                    d2ût[i,j,k] *= inv(X[i] + Y[j] + Z[k])
                end
            end
        end
        # X = Y = Z = 0. Solution is not unique. Set mean to 0
        d2ût[1,1,1] = 0
    end

    # inverse fft
    if _rfft
        u = mul!(u, iP, d2û)
    else
        d2û = iP*d2û
        u = tmap!(real, u, d2û)
    end

    return u
end

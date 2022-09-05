#####
##### Dipole kernel
#####

"""
    dipole_kernel(sz, vsz; kwargs...) =
        dipole_kernel(Float64, sz, vsz; kwargs...)

    dipole_kernel(
        ::Type{T<:AbstractFloat},
        sz::NTuple{3, Integer},
        vsz::NTuple{3, Real};
        bdir::NTuple{3, Real} = (0, 0, 1),
        method::Symbol = :kspace,
        dsz::NTuple{3, Integer} = sz,
        transform::Union{Nothing, Symbol} = nothing,
        shift::Bool = false
    ) -> Array{T, 3}

Dipole kernel.

By default the dipole kernel is constructed in k-space and centered at index `(1,1,1)`.

### Arguments
- `sz::NTuple{3, Integer}`: array size
- `vsz::NTuple{3, Real}`: voxel size

### Keywords
- `bdir::NTuple{3, Real} = (0, 0, 1)`: unit vector of B field direction
- `method::Symbol = :kspace`: create in image space `(:i, :ispace)` or
    kspace `(:k, :kspace)`
- `dsz::NTuple{3, Integer} = sz`:
    - `method ∈ (:k, :kspace)`: unused
    - `method ∈ (:i, :ispace)`: dipole kernel size in image space
- `transform::Union{Nothing, Symbol} = nothing`:
    - `method ∈ (:k, :kspace)`: create `:rfft` or `:fft` kspace dipole kernel
    - `method ∈ (:i, :ispace)`: transform image space dipole kernel `(:rfft, :fft)`
- `shift::Bool = false`:
    - `method ∈ (:k, :kspace)`:
        kernel centered at `(1,1,1)` (`false`) or `N÷2+1` (`true`)
    - `method ∈ (:i, :ispace) and transform = nothing`:
        kernel centered at `N÷2+1` (`false`) or `(1,1,1)` (`true`)
    - `method ∈ (:i, :ispace) and transform ∈ (:rfft, :fft)`:
        kernel centered at `(1,1,1)` (`false`) or `N÷2+1` (`true`)

### Returns
- `Array{T, 3}`: dipole kernel
"""
dipole_kernel(sz, vsz; kwargs...) =
    dipole_kernel(Float64, sz, vsz; kwargs...)

function dipole_kernel(
    ::Type{T},
    sz::NTuple{3, Integer},
    vsz::NTuple{3, Real};
    bdir::NTuple{3, Real} = (0, 0, 1),
    method::Symbol = :kspace,
    dsz::NTuple{3, Integer} = sz,
    transform::Union{Nothing, Symbol} = nothing,
    shift::Bool = false
) where {T<:AbstractFloat}
    all(>(0), sz)  || throw(ArgumentError("invalid array size"))
    all(>(0), vsz) || throw(DomainError(vsz, "vsz must be > 0"))
    all(>(0), dsz) || throw(DomainError(dsz, "dsz must be > 0"))
    norm(bdir) > 0 || throw(DomainError(bdir, "bdir must not be zero"))

    checkopts(method, (:k, :kspace, :i, :ispace), :method)
    transform !== nothing && checkopts(transform, (:fft, :rfft), :transform)

    if method == :k || method == :kspace
        D = Array{T, 3}(undef, transform == :rfft ? (sz[1]>>1 + 1, sz[2], sz[3]) : sz)
        return _dipole_kernel!(D, sz, vsz, bdir, :k; shift=shift)

    elseif method == :i || method == :ispace
        if transform === nothing
            d = Array{T, 3}(undef, sz)
            return _dipole_kernel!(d, dsz, vsz, bdir, :i; shift=shift)

        elseif transform == :fft
            FFTW.set_num_threads(FFTW_NTHREADS[])
            D = Array{T, 3}(undef, sz)
            D̂ = Array{complex(T), 3}(undef, sz)
            P = plan_fft!(D̂)

            D = _dipole_kernel!(D, D̂, dsz, vsz, bdir, P)
            return shift ? fftshift(D) : D

        elseif transform == :rfft
            FFTW.set_num_threads(FFTW_NTHREADS[])
            D = Array{T, 3}(undef, (sz[1]>>1 + 1, sz[2], sz[3]))
            D̂ = Array{complex(T), 3}(undef, (sz[1]>>1 + 1, sz[2], sz[3]))
            d = Array{T, 3}(undef, sz)
            P = plan_rfft(d)

            D = _dipole_kernel!(D, D̂, d, dsz, vsz, bdir, P)
            return shift ? fftshift(D) : D
        end
    end

    return nothing
end

function _dipole_kernel!(
    D::AbstractArray{T, 3},
    D̂::AbstractArray{Complex{T}, 3},
    d::AbstractArray{T, 3},
    sz::NTuple{3, Integer},
    vsz::NTuple{3, Real},
    bdir::NTuple{3, Real},
    P::Union{FFTW.rFFTWPlan{T, -1}, FFTW.cFFTWPlan{Complex{T}, -1}},
    method::Symbol,
    transform::Symbol,
) where {T<:AbstractFloat}
    if method == :k
        D = _dipole_kernel!(D, size(d), vsz, bdir, method)
    elseif method == :i && transform == :fft
        D = _dipole_kernel!(D, D̂, sz, vsz, bdir, P)
    elseif method == :i && transform == :rfft
        D = _dipole_kernel!(D, D̂, d, sz, vsz, bdir, P)
    end
    return D
end

function _dipole_kernel!(
    D::AbstractArray{T, 3},
    D̂::AbstractArray{Complex{T}, 3},
    d::AbstractArray{T, 3},
    sz::NTuple{3, Integer},
    vsz::NTuple{3, Real},
    bdir::NTuple{3, Real},
    P::Union{FFTW.rFFTWPlan{T, -1}, FFTW.cFFTWPlan{Complex{T}, -1, false}}
) where {T<:AbstractFloat}
    d = _dipole_kernel!(d, sz, vsz, bdir, :i, shift=true)

    D̂ = mul!(D̂, P, d)
    D = tmap!(real, D, D̂)

    return D
end

function _dipole_kernel!(
    D::AbstractArray{T, 3},
    D̂::AbstractArray{Complex{T}, 3},
    sz::NTuple{3, Integer},
    vsz::NTuple{3, Real},
    bdir::NTuple{3, Real},
    P::FFTW.cFFTWPlan{Complex{T}, -1, true}
) where {T<:AbstractFloat}
    D̂ = _dipole_kernel!(D̂, sz, vsz, bdir, :i, shift=true)

    D̂ = P*D̂
    D = tmap!(real, D, D̂)

    return D
end

function _dipole_kernel!(
    D::AbstractArray{T, 3},
    sz::NTuple{3, Integer},
    vsz::NTuple{3, Real},
    bdir::NTuple{3, Real},
    method::Symbol;
    shift::Bool = false,
) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    nx, ny, nz = size(D)

    dx = convert(NTuple{3, Float64}, vsz)
    B = SVector{3, Float64}(bdir)

    if !(norm(B) ≈ 1)
        B = B ./ norm(B)
    end

    if method == :k
        X, Y, Z = _grid(sz, dx, fft=true, shift=shift)
        a = 1 / 3

        @batch for k in 1:nz
            for j in 1:ny
                for i in 1:nx
                    k̂ = SVector{3, Float64}(X[i], Y[j], Z[k])
                    kz = k̂⋅B
                    k2 = k̂⋅k̂
                    D[i,j,k] = a - (kz*kz)/k2
                end
            end
        end
        # set DC term (k̂⋅k̂ := 0)
        D[1] = 0

    elseif method == :i
        dsz = map(n -> iseven(n) ? n-1 : n, sz)
        Xd, Yd, Zd = _grid(dsz, dx)
        X1, X2 = extrema(Xd)
        Y1, Y2 = extrema(Yd)
        Z1, Z2 = extrema(Zd)

        X, Y, Z = _grid(size(D), dx, shift=shift)
        a = prod(dx) * inv(4*pi)

        @batch for k in 1:nz
            z = Z[k]
            zout = z < Z1 || z > Z2
            for j in 1:ny
                y = Y[j]
                zyout = zout || y < Y1 || y > Y2
                for i in 1:nx
                    x = X[i]
                    zyxout = zyout || x < X1 || x > X2

                    if zyxout
                        D[i,j,k] = 0
                    else
                        r = SVector{3, Float64}(x, y, z)
                        rz = r⋅B
                        r2 = r⋅r
                        @fastpow D[i,j,k] = a * (3*rz^2-r2)/sqrt(r2^5)
                    end
                end
            end
        end
        # Lorentz sphere correction
        D[1] = 0
    end

    return D
end


#####
##### Spherical Mean Value kernel
#####

"""
    smv_kernel(sz, vsz, r; kwargs...) =
        smv_kernel(Float64, sz, vsz, r; kwargs...)

    smv_kernel(
        ::Type{T<:AbstractFloat},
        sz::NTuple{3, Integer},
        vsz::NTuple{3, Real}
        r::Real;
        transform::Union{Nothing, Symbol} = nothing,
        shift::Bool = false
    ) -> Array{T, 3}

Spherical mean value kernel (SMV).

### Arguments
- `sz::NTuple{3, Integer}`: array size
- `vsz::NTuple{3, Real}`: voxel size
- `r::Real`: radius of sphere in units of `vsz`

### Keywords
- `transform::Union{Nothing, Symbol} = nothing`:
    transform SMV kernel `(:rfft, :fft)`
- `shift::Bool = false`:
    - `transform = nothing`:
        sphere centered at `N÷2+1` (`false`) or `(1,1,1)` (`true`)
    - `transform ∈ (:rfft, :fft)`:
        sphere centered at `(1,1,1)` (`false`) or `N÷2+1` (`true`)

### Returns
- `Array{T, 3}`: SMV kernel
"""
smv_kernel(sz, vsz, r; kwargs...) =
    smv_kernel(Float64, sz, vsz, r; kwargs...)

function smv_kernel(
    ::Type{T},
    sz::NTuple{3, Integer},
    vsz::NTuple{3, Real},
    r::Real;
    transform::Union{Nothing, Symbol} = nothing,
    shift::Bool = false
) where {T<:AbstractFloat}
    all(>(0), sz)  || throw(ArgumentError("invalid array size"))
    all(>(0), vsz) || throw(DomainError(vsz, "vsz must be > 0"))
    transform !== nothing && checkopts(transform, (:fft, :rfft), :transform)

    if transform === nothing
        s = Array{T, 3}(undef, sz)
        return _smv_kernel!(s, vsz, r, shift=shift)

    elseif transform == :fft
        FFTW.set_num_threads(FFTW_NTHREADS[])
        S = Array{T, 3}(undef, sz)
        Ŝ = Array{complex(T), 3}(undef, sz)
        P = plan_fft!(Ŝ)

        S  = _smv_kernel!(S, Ŝ, vsz, r, P)
        return shift ? fftshift(S) : S

    elseif transform == :rfft
        FFTW.set_num_threads(FFTW_NTHREADS[])
        S = Array{T, 3}(undef, (sz[1]>>1 + 1, sz[2], sz[3]))
        Ŝ = Array{complex(T), 3}(undef, (sz[1]>>1 + 1, sz[2], sz[3]))
        s = Array{T, 3}(undef, sz)
        P = plan_rfft(s)

        S = _smv_kernel!(S, Ŝ, s, vsz, r, P)
        return shift ? fftshift(S) : S
    end

    return nothing
end

function _smv_kernel!(
    s::AbstractArray{T, 3},
    vsz::NTuple{3, Real},
    r::Real;
    shift::Bool = false
) where {T<:AbstractFloat}
    s = _sphere!(s, vsz, r, shift=shift)

    # normalizing
    a = inv(sum(s))
    s = tmap!(x -> a*x, s)

    return s
end

function _smv_kernel!(
    S::AbstractArray{T, 3},
    Ŝ::AbstractArray{Complex{T}, 3},
    s::AbstractArray{T, 3},
    vsz::NTuple{3, Real},
    r::Real,
    P::Union{FFTW.rFFTWPlan{T, -1}, FFTW.cFFTWPlan{Complex{T}, -1, false}}
) where {T<:AbstractFloat}
    s = _sphere!(s, vsz, r, shift=true)

    # normalization factor
    a = inv(sum(s))

    # fft, discard imaginary (even function -> imag = 0), and normalize
    Ŝ = mul!(Ŝ, P, s)
    S = tmap!(x -> a*real(x), S, Ŝ)

    return S
end

function _smv_kernel!(
    S::AbstractArray{T, 3},
    Ŝ::AbstractArray{Complex{T}, 3},
    vsz::NTuple{3, Real},
    r::Real,
    P::FFTW.cFFTWPlan{Complex{T}, -1, true}
) where {T<:AbstractFloat}
    Ŝ = _sphere!(Ŝ, vsz, r, shift=true)

    # normalization factor
    a = inv(sum(Ŝ))

    # fft, discard imaginary (even function -> imag = 0), and normalize
    Ŝ = P*Ŝ
    S = tmap!(x -> a*real(x), S, Ŝ)

    return S
end

function _sphere!(
    s::AbstractArray{T, 3},
    vsz::NTuple{3, Real},
    r::Real;
    shift::Bool = false
) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    sz = size(s)
    nx, ny, nz = sz

    dx = convert(NTuple{3, Float64}, vsz)
    r2 = convert(Float64, r*r)

    X, Y, Z = _grid(sz, dx, shift=shift)

    zeroT = zero(T)
    oneT = one(T)

    @inbounds for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                R = SVector{3, Float64}(X[i], Y[j], Z[k])
                s[i,j,k] = ifelse(R⋅R <= r2, oneT, zeroT)
            end
        end
    end

    return s
end


#####
##### Laplace kernel
#####

"""
    laplace_kernel(vsz) =
        laplace_kernel(Float64, vsz)

    laplace_kernel(sz, vsz; kwargs...) =
        laplace_kernel(Float64, sz, vsz; kwargs...)

    laplace_kernel(::Type{T<:AbstractFloat}, vsz::NTuple{3, Real}) =
        laplace_kernel(T, (3, 3, 3), vsz, transform=nothing, shift=false)

    laplace_kernel(
        ::Type{T<:AbstractFloat},
        sz::NTuple{3, Integer},
        vsz::NTuple{3, Real};
        negative::Bool = false,
        transform::Union{Nothing, Symbol} = nothing,
        shift::Bool = false
    ) -> Array{T, 3}

Discrete 7-point stencil Laplacian kernel.

### Arguments
- `sz::NTuple{3, Integer}`: array size
- `vsz::NTuple{3, Real}`: voxel size

### Keywords
- `negative::Bool = false`: construct negative Laplacian (`true`)
- `transform::Union{Nothing, Symbol} = nothing`:
    transform Laplacian kernel `(:rfft, :fft)`
- `shift::Bool = false`:
    - `transform = nothing`:
        sphere centered at `N÷2+1` (`false`) or `(1,1,1)` (`true`)
    - `transform ∈ (:rfft, :fft)`:
        sphere centered at `(1,1,1)` (`false`) or `N÷2+1` (`true`)

### Returns
- `Array{T, 3}`: Laplacian kernel
"""
laplace_kernel(vsz) =
    laplace_kernel(Float64, vsz)

laplace_kernel(sz, vsz; kwargs...) =
    laplace_kernel(Float64, sz, vsz; kwargs...)

laplace_kernel(::Type{T}, vsz::NTuple{3, Real}) where {T<:AbstractFloat} =
    laplace_kernel(T, (3, 3, 3), vsz, transform=nothing, shift=false)

function laplace_kernel(
    ::Type{T},
    sz::NTuple{3, Integer},
    vsz::NTuple{3, Real};
    negative::Bool = false,
    transform::Union{Nothing, Symbol} = nothing,
    shift::Bool = false
) where {T<:AbstractFloat}
    all(>(0), sz)  || throw(ArgumentError("invalid array size"))
    all(>(0), vsz) || throw(DomainError(vsz, "vsz must be > 0"))
    transform !== nothing && checkopts(transform, (:fft, :rfft), :transform)

    if transform === nothing
        Δ = Array{T, 3}(undef, sz)
        return _laplace_kernel!(Δ, vsz, negative=negative, shift=shift)

    elseif transform == :fft
        FFTW.set_num_threads(FFTW_NTHREADS[])
        L = Array{T, 3}(undef, sz)
        L̂ = Array{complex(T), 3}(undef, sz)
        P = plan_fft!(L̂)

        L = _laplace_kernel!(L, L̂, vsz, P, negative=negative)
        return shift ? fftshift(L) : L

    elseif transform == :rfft
        FFTW.set_num_threads(FFTW_NTHREADS[])
        L = Array{T, 3}(undef, (sz[1]>>1 + 1, sz[2], sz[3]))
        L̂ = Array{complex(T), 3}(undef, (sz[1]>>1 + 1, sz[2], sz[3]))
        Δ = Array{T, 3}(undef, sz)
        P = plan_rfft(Δ)

        L = _laplace_kernel!(L, L̂, Δ, vsz, P, negative=negative)
        return shift ? fftshift(L) : L
    end

    return nothing
end

function _laplace_kernel!(
    L::AbstractArray{T, 3},
    L̂::AbstractArray{Complex{T}, 3},
    Δ::AbstractArray{T, 3},
    vsz::NTuple{3, Real},
    P::Union{FFTW.rFFTWPlan{T, -1}, FFTW.cFFTWPlan{Complex{T}, -1, false}};
    negative::Bool = false,
) where {T<:AbstractFloat}
    Δ = _laplace_kernel!(Δ, vsz, negative=negative, shift=true)

    L̂ = mul!(L̂, P, Δ)
    L = tmap!(real, L, L̂)

    return L
end

function _laplace_kernel!(
    L::AbstractArray{T, 3},
    L̂::AbstractArray{Complex{T}, 3},
    vsz::NTuple{3, Real},
    P::FFTW.cFFTWPlan{Complex{T}, -1, true};
    negative::Bool = false
) where {T<:AbstractFloat}
    L̂ = _laplace_kernel!(L̂, vsz, negative=negative, shift=true)

    L̂ = P*L̂
    L = tmap!(real, L, L̂)

    return L
end

function _laplace_kernel!(
    Δ::AbstractArray{T, 3},
    vsz::NTuple{3, Real};
    negative::Bool = false,
    shift::Bool = false,
) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    hx, hy, hz = convert(NTuple{3, Float64}, inv.(vsz.*vsz))
    D = -2*(hx + hy + hz)

    if negative
        D = -D
        hx = -hx
        hy = -hy
        hz = -hz
    end

    Δ = tfill!(Δ, zero(T))

    if shift
        Δ[1,1,1]   = D
        Δ[2,1,1]   = hx
        Δ[end,1,1] = hx
        Δ[1,2,1]   = hy
        Δ[1,end,1] = hy
        Δ[1,1,2]   = hz
        Δ[1,1,end] = hz
    else
        i, j, k = size(Δ).>>1 .+ 1
        Δ[i,j,k-1] = hz
        Δ[i,j-1,k] = hy
        Δ[i-1,j,k] = hx
        Δ[i,j,k]   = D
        Δ[i+1,j,k] = hx
        Δ[i,j+1,k] = hy
        Δ[i,j,k+1] = hz
    end

    return Δ
end


#####
##### Utility
#####

function _grid(
    sz::NTuple{N, Integer},
    dx::NTuple{N, Real};
    fft::Bool = false,
    shift::Bool = false
) where {N}
    return ntuple(Val(N)) do i
        if fft
            # AbstractFFTs.fftfreq(n, fs) = Frequencies((n+1) >> 1, n, fs/n)
            X = Frequencies((sz[i]+1) >> 1, sz[i], inv(sz[i]*dx[i]))
            collect(shift ? fftshift(X) : X)
        else
            X = Frequencies((sz[i]+1) >> 1, sz[i], dx[i])
            collect(shift ? X : fftshift(X))
        end
    end
end

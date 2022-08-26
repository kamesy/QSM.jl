"""
    nltv(
        f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        W::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, N)}} = nothing,
        Wtv::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, N)}} = nothing,
        pad::NTuple{3, Integer} = (0, 0, 0),
        Dkernel::Symbol = :k,
        bdir::NTuple{3, Real} = (0, 0, 1),
        lambda::Real = 1e-3,
        rho::Real = 100*lambda,
        mu::Real = 1,
        tol::Real = 1e-3,
        maxit::Integer = 250,
        toln::Real = 1e-6,
        maxitn::Integer = 10,
        verbose::Bool = false,
    ) -> typeof(similar(f))

Nonlinear total variation deconvolution using ADMM [1].

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) local
    field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size

### Keywords
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, N)}} = nothing`:
    data fidelity weights
- `Wtv::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, 5)}} = nothing`:
    total variation weights
    - `M = 3`: same weights for all three gradient directions and all echoes
    - `M = 4 = N`: same weights for all three gradient directions, different weights for echoes
    - `M = 5, (size(Wtv)[4,5] = [1 or N, 3]`: different weights for each gradient direction
- `pad::NTuple{3, Integer} = (0, 0, 0)`: zero padding array
    - `< 0`: no padding
    - `≥ 0`: minimum padding to fast fft size
- `bdir::NTuple{3, Real} = (0, 0, 1)`: unit vector of B field direction
- `Dkernel::Symbol = :k`: dipole kernel method
- `lambda::Real = 1e-3`: regularization parameter
- `rho::Real = 100*lambda`: Lagrange multiplier penalty parameter
- `mu::Real = 1`: Lagrange multiplier penalty parameter (unused if `W = nothing`)
- `tol::Real = 1e-3`: stopping tolerance
- `maxit::Integer = 250`: maximum number of iterations
- `toln::Real = 1e-6`: stopping tolerance for Newton method
- `maxitn::Integer = 10`: maximum number of iterations for Newton method
- `verbose::Bool = false`: print convergence information

### Returns
- `typeof(similar(f))`: susceptibility map

### References
[1] Milovic C, Bilgic B, Zhao B, Acosta‐Cabronero J, Tejos C. Fast nonlinear
    susceptibility inversion with variational regularization.
    Magnetic resonance in medicine. 2018 Aug;80(2):814-21.
"""
function nltv(
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    W::Union{Nothing, AbstractArray{<:AbstractFloat}} = nothing,
    Wtv::Union{Nothing, AbstractArray{<:AbstractFloat}} = nothing,
    pad::NTuple{3, Integer} = (0, 0, 0),
    Dkernel::Symbol = :k,
    bdir::NTuple{3, Real} = (0, 0, 1),
    lambda::Real = 1e-3,
    rho::Real = 100*lambda,
    mu::Real = 1,
    tol::Real = 1e-3,
    maxit::Integer = 250,
    toln::Real = 1e-6,
    maxitn::Integer = 10,
    verbose::Bool = false,
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _nltv!(
        tzero(f), f, mask, vsz, W, Wtv, pad, Dkernel, bdir,lambda,
        rho, mu, tol, maxit, toln, maxitn, verbose
    )
end

function _nltv!(
    x::AbstractArray{<:AbstractFloat, N},
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    W::Union{Nothing, AbstractArray{<:AbstractFloat}},
    Wtv::Union{Nothing, AbstractArray{<:AbstractFloat}},
    pad::NTuple{3, Integer},
    Dkernel::Symbol,
    bdir::NTuple{3, Real},
    lambda::Real,
    rho::Real,
    mu::Real,
    tol::Real,
    maxit::Integer,
    toln::Real,
    maxitn::Integer,
    verbose::Bool,
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    checkshape(x, f, (:x, :f))
    checkshape(axes(mask), axes(f)[1:3], (:mask, :f))

    if W !== nothing
        checkshape(Bool, axes(W), axes(f)[1:3]) ||
        checkshape(W, f, (:W, :f))
    end

    if Wtv !== nothing
        if ndims(Wtv) < 5
            checkshape(Bool, axes(Wtv), axes(f)[1:3]) ||
            checkshape(Wtv, f, (:Wtv, :f))
        else
            checkshape(Bool, axes(Wtv), (axes(f)[1:3]..., 1:1, 1:3)) ||
            checkshape(Bool, axes(Wtv), (axes(f)[1:3]..., axes(f, 4), 1:3)) ||
            checkshape(Wtv, f, (:Wtv, :f))
        end
    end

    checkopts(Dkernel, (:k, :kspace, :i, :ispace), :Dkernel)

    # convert scalars
    zeroT = zero(T)
    oneT = one(T)

    ρ = convert(T, rho)
    μ = convert(T, mu)
    λ = convert(T, lambda)
    ϵ = convert(T, tol)
    ϵn = convert(T, toln*toln)

    iρ = inv(ρ)
    iμ = inv(μ)
    λiρ = λ*iρ

    # pad to fast fft size
    xp = padfastfft(@view(f[:,:,:,1]), pad, rfft=true)
    m = padfastfft(mask, pad, rfft=true)

    # initialize variables and fft
    sz0 = size(mask)
    sz  = size(m)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    fp = similar(xp)
    x0 = similar(xp)

    D  = Array{T}(undef, sz_)           # dipole kernel
    L  = Array{T}(undef, sz_)           # laplace kernel
    iA = Array{T}(undef, sz_)           # 1 / (D'D - rho*Δ)

    X̂ = Array{complex(T)}(undef, sz_)   # in-place rfft var
    F̂ = Array{complex(T)}(undef, sz_)   # pre-computed rhs

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(xp)
    iP = inv(P)

    # admm
    ux = similar(xp)
    uy = similar(xp)
    uz = similar(xp)

    dx = similar(xp)
    dy = similar(xp)
    dz = similar(xp)

    ue = similar(xp)
    ze = similar(xp)

    if W !== nothing
        W2 = similar(xp)
    end

    if Wtv !== nothing
        if ndims(Wtv) == 3
            λiρWx = padarray!(similar(xp), Wtv)
            λiρWy = λiρWx
            λiρWz = λiρWx
        elseif ndims(Wtv) == 4
            λiρWx = similar(xp)
            λiρWy = λiρWx
            λiρWz = λiρWx
        elseif size(Wtv, 4) == 1
            λiρWx = padarray!(similar(xp), @view(Wtv[:,:,:,1,1]))
            λiρWx = padarray!(similar(xp), @view(Wtv[:,:,:,1,2]))
            λiρWx = padarray!(similar(xp), @view(Wtv[:,:,:,1,3]))
        else
            λiρWx = similar(xp)
            λiρWy = similar(xp)
            λiρWz = similar(xp)
        end
    end

    # get kernels
    D = _dipole_kernel!(D, F̂, xp, sz0, vsz, bdir, P, Dkernel, :rfft)
    L = _laplace_kernel!(L, F̂, xp, vsz, P, negative=true)

    for t in axes(f, 4)
        if verbose && size(f, 4) > 1
            @printf("Echo: %d/%d\n", t, size(f, 4))
        end

        fp = padarray!(fp, @view(f[:, :, :, t]))

        # iA = (μ*D^H*D - ρ*Δ)^-1
        @batch for I in eachindex(iA)
            a = μ*conj(D[I])*D[I] + ρ*L[I]
            iA[I] = ifelse(iszero(a), zeroT, inv(a))
        end

        F̂ = tfill!(F̂, 0)

        if W !== nothing
            W2 = padarray!(W2, @view(W[:,:,:,min(t, end)]))
            @batch for I in eachindex(W2)
                W2[I] *= iμ*W2[I]
            end
        end

        # total variation weights
        if Wtv !== nothing
            if ndims(Wtv) == 4
                λiρWx = padarray!(λiρWx, @view(Wtv[:,:,:,t]))
            elseif ndims(Wtv) == 5 && size(Wtv, 4) > 1
                λiρWx = padarray!(λiρWx, @view(Wtv[:,:,:,t,1]))
                λiρWy = padarray!(λiρWy, @view(Wtv[:,:,:,t,2]))
                λiρWz = padarray!(λiρWz, @view(Wtv[:,:,:,t,3]))
            end
        end

        ux = tfill!(ux, zeroT)
        uy = tfill!(uy, zeroT)
        uz = tfill!(uz, zeroT)
        dx, dy, dz = _gradfp!(dx, dy, dz, fp, vsz)

        if verbose
            @printf("\n iter\t  ||x-xprev||/||x||\n")
        end

        for i in 1:maxit
            x0, xp = xp, x0

            ##################################################################
            # x - subproblem
            ##################################################################
            # d = z - u
            xp = _gradfp_adj!(xp, dx, dy, dz, vsz)

            X̂ = mul!(X̂, P, xp)

            @batch for I in eachindex(X̂)
                X̂[I] *= ρ*iA[I]
                X̂[I] += F̂[I]
            end

            if W !== nothing
                F̂ = tcopyto!(F̂, X̂) # real ifft overwrites input
            end

            xp = mul!(xp, iP, X̂)

            ##################################################################
            # convergence check
            ##################################################################
            # [1] ndx = norm(x - xprev)
            # [2] nx  = norm(x)
            @batch threadlocal=zeros(T, 2)::Vector{T} for I in eachindex(xp)
                a, b = xp[I], x0[I]
                threadlocal[1] = muladd(a-b, a-b, threadlocal[1])
                threadlocal[2] = muladd(a, a, threadlocal[2])
            end
            ndx, nx = sqrt.(sum(threadlocal::Vector{Vector{T}}))

            if verbose
                @printf("%3d/%d\t    %.4e\n", i, maxit, ndx/nx)
            end

            if ndx < ϵ*nx || i == maxit
                break
            end

            ##################################################################
            # z - subproblem and Lagrange multiplier update
            ##################################################################
            dx, dy, dz = _gradfp!(dx, dy, dz, xp, vsz)

            # z = shrink(∇x + u, λ/ρ)   z-subproblem
            # u = u + ∇x - z            Lagrange multiplier update
            # d = z - u                 pre-compute for x-problem
            if Wtv !== nothing
                @batch for I in eachindex(ux)
                    λiρ = λiρWx[I]
                    u = ux[I] + dx[I]
                    z = ifelse(abs(u) ≤ λiρ, zeroT, copysign(abs(u)-λiρ, u))
                    ux[I] = u - z
                    dx[I] = z - u + z
                end

                @batch for I in eachindex(uy)
                    λiρ = λiρWy[I]
                    u = uy[I] + dy[I]
                    z = ifelse(abs(u) ≤ λiρ, zeroT, copysign(abs(u)-λiρ, u))
                    uy[I] = u - z
                    dy[I] = z - u + z
                end

                @batch for I in eachindex(uz)
                    λiρ = λiρWz[I]
                    u = uz[I] + dz[I]
                    z = ifelse(abs(u) ≤ λiρ, zeroT, copysign(abs(u)-λiρ, u))
                    uz[I] = u - z
                    dz[I] = z - u + z
                end

            else
                @batch for I in eachindex(ux)
                    u = ux[I] + dx[I]
                    z = ifelse(abs(u) ≤ λiρ, zeroT, copysign(abs(u)-λiρ, u))
                    ux[I] = u - z
                    dx[I] = z - u + z
                end

                @batch for I in eachindex(uy)
                    u = uy[I] + dy[I]
                    z = ifelse(abs(u) ≤ λiρ, zeroT, copysign(abs(u)-λiρ, u))
                    uy[I] = u - z
                    dy[I] = z - u + z
                end

                @batch for I in eachindex(uz)
                    u = uz[I] + dz[I]
                    z = ifelse(abs(u) ≤ λiρ, zeroT, copysign(abs(u)-λiρ, u))
                    uz[I] = u - z
                    dz[I] = z - u + z
                end
            end

            ##################################################################
            # F̂ - update
            ##################################################################
            # z = z - (W^2*sin(z-f) + μ*z - μ*(D*x+u)) / (W^2*cos(z-f) + μ)
            #     z - (W^2/μ*sin(z-f) + z - (D*x+u)) / (W^2/μ*cos(z-f) + 1)
            # u = u + D*x - z
            # F̂ = μ*D^H*(z - u) / (μ*D^H*D - ρ*Δ)
            @batch for I in eachindex(F̂)
                F̂[I] *= D[I]
            end

            ze = mul!(ze, iP, F̂)
            @batch for I in eachindex(ze)
                a = ze[I] + ue[I]
                ze[I] = a
                ue[I] = a
            end

            if W !== nothing
                for _ in 1:maxitn
                    @batch threadlocal=zeros(T, 2)::Vector{T} for I in eachindex(ue)
                        u = ue[I]
                        w = W2[I]
                        z = ze[I]
                        φ = fp[I]

                        s, c = sincos_fast(z - φ)
                        δ = (muladd(w, s, z) - u) / muladd(w, c, oneT)

                        ze[I] = z - δ
                        threadlocal[1] = muladd(δ, δ, threadlocal[1])
                        threadlocal[2] = muladd(z, z, threadlocal[2])
                    end

                    nδ, nz = sum(threadlocal::Vector{Vector{T}})

                    if nδ < nz*ϵn
                        break
                    end
                end

            else
                for _ in 1:maxitn
                    @batch threadlocal=zeros(T, 2)::Vector{T} for I in eachindex(ue)
                        u = ue[I]
                        z = ze[I]
                        φ = fp[I]

                        s, c = sincos_fast(z - φ)
                        δ = (s + μ*(z - u)) / (c + μ)

                        ze[I] = z - δ
                        threadlocal[1] = muladd(δ, δ, threadlocal[1])
                        threadlocal[2] = muladd(z, z, threadlocal[2])
                    end

                    nδ, nz = sum(threadlocal::Vector{Vector{T}})

                    if nδ < nz*ϵn
                        break
                    end
                end
            end

            @batch for I in eachindex(ue)
                u = ue[I]
                z = ze[I]
                ue[I] = u - z
                x0[I] = z - u + z
            end

            F̂ = mul!(F̂, P, x0)
            @batch for I in eachindex(F̂)
                F̂[I] *= μ*conj(D[I])*iA[I]
            end
        end

        unpadarray!(@view(x[:,:,:,t]), xp)

        if verbose
            println()
        end

    end

    return x
end

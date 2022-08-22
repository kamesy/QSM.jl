"""
    rts(
        f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        pad::NTuple{3, Integer} = (0, 0, 0),
        Dkernel::Symbol = :k,
        bdir::NTuple{3, Real} = (0, 0, 1),
        lstol::Integer = 4,
        delta::Real = 0.15,
        mu::Real = 1e5,
        rho::Real = 10,
        tol::Real = 1e-2,
        maxit::Integer = 20,
        verbose::Bool = false,
    ) -> typeof(similar(f))

Rapid two-step dipole inversion with sparsity priors [1].

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) local
    field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size

### Keywords
- `pad::NTuple{3, Integer} = (0, 0, 0)`: zero padding array
    - `< 0`: no padding
    - `≥ 0`: minimum padding to fast fft size
- `bdir::NTuple{3, Real} = (0, 0, 1)`: unit vector of B field direction
- `Dkernel::Symbol = :k`: dipole kernel method
- `lstol::Integer = 4`: stopping tolerance (# of iterations) for lsmr solver
- `delta::Real = 0.15`: threshold for ill-conditioned k-space region
- `mu::Real = 1e5`: regularization parameter for tv minimization
- `rho::Real = 10`: Lagrange multiplier penalty parameter
- `tol::Real = 1e-2`: stopping tolerance
- `maxit::Integer = 20`: maximum number of iterations
- `verbose::Bool = false`: print convergence information

### Returns
- `typeof(similar(f))`: susceptibility map

### References
[1] Kames C, Wiggermann V, Rauscher A. Rapid two-step dipole inversion for
    susceptibility mapping with sparsity priors.
    Neuroimage. 2018 Feb 15;167:276-83.
"""
function rts(
    f::AbstractArray{<:AbstractFloat, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    pad::NTuple{3, Integer} = (0, 0, 0),
    Dkernel::Symbol = :k,
    bdir::NTuple{3, Real} = (0, 0, 1),
    lstol::Integer = 4,
    delta::Real = 0.15,
    mu::Real = 1e5,
    rho::Real = 10,
    tol::Real = 1e-2,
    maxit::Integer = 20,
    verbose::Bool = false,
    tau::Real = 0.7,
    gamma::Real = 5,
) where {N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _rts!(
        tzero(f), f, mask, vsz, pad, Dkernel, bdir, lstol, delta, mu, rho,
        tol, maxit, verbose, tau, gamma
    )
end

function _rts!(
    x::AbstractArray{<:AbstractFloat, N},
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    pad::NTuple{3, Integer},
    Dkernel::Symbol,
    bdir::NTuple{3, Real},
    lstol::Integer,
    delta::Real,
    mu::Real,
    rho::Real,
    tol::Real,
    maxit::Integer,
    verbose::Bool,
    tau::Real,
    gamma::Real,
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    checkshape(x, f, (:x, :f))
    checkshape(axes(mask), axes(f)[1:3])

    Dkernel ∈ (:k, :kspace, :i, :ispace) ||
        throw(ArgumentError("Dkernel must be one of :k, :kspace, :i, :ispace, got :$(Dkernel)"))

    # convert scalars
    _zero = zero(T)

    ρ = convert(T, rho)
    δ = convert(T, delta)
    μ = convert(T, mu)
    γ = convert(T, gamma)
    τ = convert(T, tau)
    ϵ = convert(T, tol)

    iγ = inv(γ)
    iρ = inv(ρ)

    # pad to fast fft size
    xp = padfastfft(@view(f[:,:,:,1]), pad, rfft=true)
    m = padfastfft(mask, pad, rfft=true)

    # initialize variables and fft
    sz0 = size(mask)
    sz  = size(m)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    x0 = similar(xp)

    D  = Array{T}(undef, sz_)   # dipole kernel
    L  = Array{T}(undef, sz_)   # laplace kernel
    M  = Array{T}(undef, sz_)   # abs(D) > δ
    iA = Array{T}(undef, sz_)   # 1 / (mu*M - rho*Δ)

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(xp)
    iP = inv(P)

    # lsmr
    A = LinearMap{complex(T)}(
        (Dv, v) -> _A_rts!(Dv, v, D),
        length(D),
        ishermitian = true,
        ismutating = true
    )

    WS = LSMRWorkspace(A)
    B̂ = reshape(WS.b, sz_)      # lsmr rhs
    X̂ = reshape(WS.x, sz_)      # lsmr solution, re-use as in-place rfft var
    F̂ = reshape(WS.v, sz_)      # re-use as pre-computed rhs for admm

    # admm
    px = similar(xp)
    py = similar(xp)
    pz = similar(xp)

    dx = similar(xp)
    dy = similar(xp)
    dz = similar(xp)

    vx = similar(xp)
    vy = similar(xp)
    vz = similar(xp)

    # get kernels
    D = _dipole_kernel!(D, F̂, xp, sz0, vsz, bdir, P, Dkernel, :rfft)
    L = _laplace_kernel!(L, F̂, xp, vsz, P, negative=true)

    # mask of well-conditioned frequencies
    @inbounds @batch for I in eachindex(M)
        M[I] = ifelse(abs(D[I]) > δ, μ, _zero)
    end

    @inbounds for t in axes(f, 4)
        if verbose && size(f, 4) > 1
            @printf("Echo: %d/%d\n", t, size(f, 4))
        end

        xp = padarray!(xp, @view(f[:, :, :, t]))

        ######################################################################
        # Step 1: Well-conditioned
        ######################################################################
        B̂ = mul!(B̂, P, xp)
        lsmr!(WS; lambda=_zero, atol=_zero, btol=_zero, maxit=lstol)

        xp = mul!(xp, iP, X̂)

        @batch for I in eachindex(xp)
            xp[I] *= m[I]
        end

        ######################################################################
        # Step 2: Ill-conditioned
        ######################################################################
        X̂ = mul!(X̂, P, xp)

        @batch for I in eachindex(F̂)
            a = muladd(ρ, L[I], M[I])
            if iszero(a)
                F̂[I] = _zero
                iA[I] = _zero
            else
                ia = inv(a)
                F̂[I] = ia * M[I] * X̂[I]
                iA[I] = ρ*ia
            end
        end

        nr = typemax(T)
        px = tfill!(px, _zero)
        py = tfill!(py, _zero)
        pz = tfill!(pz, _zero)
        vx, vy, vz = _gradfp!(vx, vy, vz, xp, vsz)

        if verbose
            @printf("\n iter\t  ||x-xprev||/||x||\n")
        end

        for i in 1:maxit
            x0, xp = xp, x0

            ##################################################################
            # x - subproblem
            ##################################################################
            # v = y - p
            xp = _gradfp_adj!(xp, vx, vy, vz, vsz)

            X̂ = mul!(X̂, P, xp)
            @batch for I in eachindex(X̂)
                X̂[I] *= iA[I]
                X̂[I] += F̂[I]
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
            # y - subproblem and Lagrange multiplier update
            ##################################################################
            dx, dy, dz = _gradfp!(dx, dy, dz, xp, vsz)

            # y = shrink(∇x + p, 1/ρ)   y-subproblem
            # p = p + ∇x - y            Lagrange multiplier update
            # v = y - p                 pre-compute for x-problem
            # d = ∇x - y                pre-compute for primal residual
            @batch for I in eachindex(px)
                d = dx[I]
                p = px[I] + d
                y = ifelse(abs(p) ≤ iρ, _zero, copysign(abs(p)-iρ, p))
                px[I] = p - y
                vx[I] = y - p + y
                dx[I] = d - y
            end

            @batch for I in eachindex(py)
                d = dy[I]
                p = py[I] + d
                y = ifelse(abs(p) ≤ iρ, _zero, copysign(abs(p)-iρ, p))
                py[I] = p - y
                vy[I] = y - p + y
                dy[I] = d - y
            end

            @batch for I in eachindex(pz)
                d = dz[I]
                p = pz[I] + d
                y = ifelse(abs(p) ≤ iρ, _zero, copysign(abs(p)-iρ, p))
                pz[I] = p - y
                vz[I] = y - p + y
                dz[I] = d - y
            end

            ##################################################################
            # parameter update
            ##################################################################
            nr0 = nr

            # nr = norm(∇x - y), primal residual
            @batch threadlocal=zero(T)::T for I in eachindex(dx)
                threadlocal += sqrt(dx[I]*dx[I] + dy[I]*dy[I] + dz[I]*dz[I])
            end
            nr = sum(threadlocal::Vector{T})

            if nr > τ*nr0
                ρ0 = ρ
                ρ *= γ
                iρ = inv(ρ)

                # update constant rhs and ilhs
                @batch for I in eachindex(iA)
                    iaprev = iA[I]
                    if !iszero(iaprev)
                        ia = inv(muladd(ρ, L[I], M[I]))
                        F̂[I] *= ia*ρ0*inv(iaprev)
                        iA[I] = ρ*ia
                    end
                end

                # update scaled dual variables
                @batch for I in eachindex(px)
                    px[I] *= iγ
                end

                @batch for I in eachindex(py)
                    py[I] *= iγ
                end

                @batch for I in eachindex(pz)
                    pz[I] *= iγ
                end
            end
        end

        unpadarray!(@view(x[:,:,:,t]), xp)

        if verbose
            println()
        end

    end

    return x
end


function _A_rts!(Dv, v, D)
    @inbounds @batch for I in eachindex(Dv)
        Dv[I] = D[I]*v[I]
    end
    return Dv
end

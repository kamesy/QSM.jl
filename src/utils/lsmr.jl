#####
##### LSMR implementation copied from IterativeSolvers.jl
##### modified to make it non-allocating and to add multi-threading
##### https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/v0.9.2/src/lsmr.jl
#####

struct LSMRWorkspace{TA, Tx, Tb, Tw}
    A::TA
    x::Tx
    b::Tb
    v::Tw
    h::Tw
    hbar::Tw
    Av::Tb
    Atu::Tw
end

function LSMRWorkspace(A)
    T = eltype(A)

    x = tfill!(Vector{T}(undef, size(A, 2)), zero(T))
    b = tfill!(Vector{T}(undef, size(A, 1)), zero(T))

    v = similar(x, T)
    h = similar(x, T)
    h̄ = similar(x, T)
    Atu = similar(x, T)
    Av = similar(b)

    return LSMRWorkspace(A, x, b, v, h, h̄, Av, Atu)
end

function LSMRWorkspace(A, b::AbstractArray)
    T = typeof(one(eltype(b)) / one(eltype(A)))
    x = similar(b, T, size(A, 2))
    return LSMRWorkspace(x, A, b)
end

function LSMRWorkspace(x::AbstractArray, A, b::AbstractArray)
    T = typeof(one(eltype(b)) / one(eltype(A)))

    u = tcopy(b)

    v = similar(x, T)
    h = similar(x, T)
    h̄ = similar(x, T)
    Atu = similar(x, T)
    Av = similar(b)

    return LSMRWorkspace(A, x, u, v, h, h̄, Av, Atu)
end


#== Not exported. Don't generate docs
"""
    lsmr(A, b; kwargs...) -> x

Same as [`lsmr!`](@ref), but allocates a solution vector `x` initialized with
zeros.
"""
==#
lsmr(A, b; kwargs...) = lsmr!(LSMRWorkspace(A, b); kwargs...)


#== Not exported. Don't generate docs
"""
    lsmr!(x, A, b; kwargs...) -> x

Minimizes ``\\|Ax - b\\|^2 + \\|λx\\|^2`` in the Euclidean norm. If multiple
solutions exists the minimum norm solution is returned.

The method is based on the Golub-Kahan bidiagonalization process. It is
algebraically equivalent to applying MINRES to the normal equations
``(A^*A + λ^2I)x = A^*b``, but has better numerical properties, especially if
``A`` is ill-conditioned.

### Arguments
- `x`: solution array
- `A`: linear operator
- `b`: right-hand side

### Keywords
- `lambda::Real = 0`: regularization parameter
- `atol::Real = 1e-6`: stopping tolerance
- `btol::Real = 1e-6`: stopping tolerance
- `conlim::Real = 1e8`: stopping tolerance
- `maxit::Int = 1 + min(size(A, 1), size(A, 2))`: maximum number of iterations
- `verbose::Bool = false`: print convergence information

### Returns
- `x`: approximated solution

### Notes
- `atol`, `btol`: LSMR continues iterations until a certain backward error
    estimate is smaller than some quantity depending on ATOL and BTOL.
    Let RES = B - A*X be the residual vector for the current approximate
    solution X.  If A*X = B seems to be consistent, LSMR terminates when
    NORM(RES) <= ATOL*NORM(A)*NORM(X) + BTOL*NORM(B).  Otherwise, LSMR
    terminates when NORM(A'*RES) <= ATOL*NORM(A)*NORM(RES).  If both tolerances
    are 1.0e-6 (say), the final NORM(RES) should be accurate to about 6 digits.
    (The final X will usually have fewer correct digits, depending on cond(A)
    and the size of LAMBDA.) If ATOL or BTOL is [], a default value of 1.0e-6
    will be used.  Ideally, they should be estimates of the relative error in
    the entries of A and B respectively.  For example, if the entries of A
    have 7 correct digits, set ATOL = 1e-7. This prevents the algorithm
    from doing unnecessary work beyond the uncertainty of the input data.
- `conlim`: LSMR terminates if an estimate of cond(A) exceeds CONLIM. For
    compatible systems Ax = b, conlim could be as large as 1.0e+12 (say).
    For least-squares problems, conlim should be less than 1.0e+8. If CONLIM
    is [], the default value is CONLIM = 1e+8. Maximum precision can be
    obtained by setting ATOL = BTOL = CONLIM = 0, but the number of iterations
    may then be excessive.

### References
    D. C.-L. Fong and M. A. Saunders,
    LSMR: An iterative algorithm for sparse least-squares problems,
    SIAM J. Sci. Comput., submitted 1 June 2010.
    See http://www.stanford.edu/~clfong/lsmr.html.
"""
==#
lsmr!(x, A, b; kwargs...) = lsmr!(LSMRWorkspace(x, A, b); kwargs...)


function lsmr!(
    L::LSMRWorkspace;
    atol::Real = 1e-6,
    btol::Real = 1e-6,
    conlim::Real = 1e8,
    maxit::Int = 1 + min(size(L.A, 1), size(L.A, 2)),
    lambda::Real = 0,
    verbose::Bool = false
)
    A, x, b = L.A, L.x, L.b
    v, h, hbar, Av, Atu = L.v, L.h, L.hbar, L.Av, L.Atu

    T = typeof(one(eltype(b)) / one(eltype(A)))
    Tr = real(T)

    At = adjoint(A)

    # form the first vectors u and v (satisfy  β*u = b,  α*v = A'u)
    u = b
    β = norm(u)

    if !iszero(β)
        u = _scal!(inv(β), u)
    end

    mul!(v, At, u)
    α = norm(v)

    if !iszero(α)
        v = _scal!(inv(α), v)
    end

    # Initialize variables for 1st iteration.
    λ = convert(Tr, lambda)

    ζbar = α * β
    αbar = α
    ρ    = one(Tr)
    ρbar = one(Tr)
    cbar = one(Tr)
    sbar = zero(Tr)

    tfill!(x, 0)
    tfill!(hbar, 0)
    tcopyto!(h, v)

    # Initialize variables for estimation of ||r||.
    βdd = β
    βd = zero(Tr)
    ρdold = one(Tr)
    τtildeold = zero(Tr)
    θtilde  = zero(Tr)
    ζ = zero(Tr)
    d = zero(Tr)

    # Initialize variables for estimation of ||A|| and cond(A).
    normA2  = abs2(α)
    maxrbar = zero(Tr)
    minrbar = typemax(Tr)

    # Items for use in stopping rules.
    i = 0
    istop = 0
    normb = β
    normr = β

    ϵa = convert(Tr, atol)
    ϵb = convert(Tr, btol)
    ϵc = conlim > 0 ? convert(Tr, inv(conlim)) : zero(Tr)

    # Exit if b = 0 or A'b = 0.
    normAr = α * β
    if iszero(normAr)
        if verbose
            @printf("The exact solution is x = 0")
        end
        return x
    end

    if verbose
        @printf("\nLSMR            Least-squares solution of  Ax = b\n")
        @printf("The matrix A has %d rows and %d cols\n", size(A, 1), size(A, 2))
        @printf("lambda = %16.10e\n", lambda)
        @printf("atol   = %8.2e\n", atol)
        @printf("btol   = %8.2e\n", btol)
        @printf("conlim = %8.2e\n", conlim)
        @printf("\n%7s %11s\t%8s %11s %5s\t%6s\t%8s\n",
            "i", "||r||", "||A'r||", "compatible", "LS", "||A||", "cond A"
        )
    end

    while i < maxit
        i += 1

        # u = A*v - α*u
        # β = norm(u)
        mul!(Av, A, v)
        u, β = _xpby_norm!(Av, -α, u)

        if !iszero(β)
            u = _scal!(inv(β), u)

            # v = A'*u - β*v
            # α = norm(v)
            mul!(Atu, At, u)
            v, α = _xpby_norm!(Atu, -β, v)

            if !iszero(α)
                v = _scal!(inv(α), v)
            end
        end

        # Construct rotation Qhat_{k,2k+1}.
        αhat = hypot(αbar, λ)
        chat = αbar / αhat
        shat = λ / αhat

        # Use a plane rotation (Q_i) to turn B_i to R_i.
        ρold = ρ
        ρ    = hypot(αhat, β)
        c    = αhat / ρ
        s    = β / ρ
        θnew = s * α
        αbar = c * α

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.
        ρbarold = ρbar
        ζold = ζ
        θbar = sbar * ρ
        ρtmp = cbar * ρ
        ρbar = hypot(ρtmp, θnew)
        cbar = ρtmp / ρbar
        sbar = θnew / ρbar
        ζ    = cbar * ζbar
        ζbar = -sbar * ζbar

        # Update h, h_hat, x.
        hbar, x, h, normx =
            _update_hbar_x_h!(hbar, x, h, v, θbar, ρ, ρold, ρbarold, ζ, ρbar, θnew)

        # Estimate of ||r||

        # Apply rotation Qhat_{k,2k+1}.
        βacute =  chat * βdd
        βcheck = -shat * βdd

        # Apply rotation Q_{k,k+1}.
        βhat =  c * βacute
        βdd  = -s * βacute

        # Apply rotation Qtilde_{k-1}.
        θtildeold = θtilde
        ρtildeold = hypot(ρdold, θbar)
        ctildeold = ρdold / ρtildeold
        stildeold = θbar / ρtildeold
        θtilde    = stildeold * ρbar
        ρdold     = ctildeold * ρbar
        βd        = -stildeold*βd + ctildeold*βhat

        τtildeold = (ζold - θtildeold * τtildeold) / ρtildeold
        τd = (ζ - θtilde * τtildeold) / ρdold

        d += abs2(βcheck)
        normr = sqrt(d + abs2(βd - τd) + abs2(βdd))

        # Estimate ||A||.
        normA2 += abs2(β)
        normA   = sqrt(normA2)
        normA2 += abs2(α)

        # Estimate cond(A).
        maxrbar = max(maxrbar, ρbarold)
        if i > 1
            minrbar = min(minrbar, ρbarold)
        end
        condA = max(maxrbar, ρtmp) / min(minrbar, ρtmp)

        # Test for convergence

        # Compute norms for convergence testing.
        normAr = abs(ζbar)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = normr / normb
        test2 = normAr / (normA * normr)
        test3 = inv(condA)

        t1 = test1 / (1 + normA * normx / normb)
        ϵr = ϵb + ϵa * normA * normx / normb

        # Allow for tolerances set by the user.
        if     test1 <= ϵr      istop = 1
        elseif test2 <= ϵa      istop = 2
        elseif test3 <= ϵc      istop = 3
        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        elseif 1 + t1    <= 1   istop = 4
        elseif 1 + test2 <= 1   istop = 5
        elseif 1 + test3 <= 1   istop = 6
        elseif i >= maxit       istop = 7
        end

        if verbose
            if size(A, 2) <= 50 ||
                maxit <= 50     ||
                i <= 10         ||
                i >= maxit-10   ||
                mod(i, 25) == 0 ||
                test3 <= 1.1*ϵc ||
                test2 <= 1.1*ϵa ||
                test1 <= 1.1*ϵr ||
                istop != 0
                @printf("[%4d/%d]", i, maxit)
                @printf("  %10.3e  %10.3e", normr, normAr)
                @printf("  %8.1e  %8.1e", test1, test2)
                @printf("  %8.1e  %8.1e\n", normA, condA)
            end
        end

        if istop > 0
            break
        end
    end

    if verbose
        @printf("\n")
    end

    return x
end


@inline function _xpby_norm!(X, b, Y)
    @. Y = muladd(b, Y, X)
    return Y, norm(Y)
end

@inline function _xpby_norm!(X::Array, b::Real, Y::Array{T}) where {T}
    Tr = real(T)
    bT = convert(Tr, b)
    @inbounds @batch threadlocal=zero(T)::T for I in eachindex(Y)
        Y[I] = muladd(bT, Y[I], X[I])
        threadlocal = muladd(conj(Y[I]), Y[I], threadlocal)
    end
    normY = sqrt(sum(real, threadlocal::Vector{T}))
    return Y, normY
end


@inline function _scal!(a, X)
    X .*= a
    return X
end

@inline function _scal!(a, X::Array{T}) where {T}
    aT = convert(real(T), a)
    @inbounds @batch for I in eachindex(X)
        X[I] *= aT
    end
    return X
end


@inline function _update_hbar_x_h!(h̄, x, h, v, θbar, ρ, ρold, ρbarold, ζ, ρbar, θnew)
    δ = -θbar * ρ / (ρold * ρbarold)
    σ = ζ / (ρ * ρbar)
    ν = -θnew / ρ

    @. h̄ = muladd(δ, h̄, h)
    @. x = muladd(σ, h̄, x)
    @. h = muladd(ν, h, v)

    return h̄, x, h, norm(x)
end

@inline function _update_hbar_x_h!(
    h̄::Array,
    x::Array{T},
    h::Array,
    v::Array,
    θbar::Real,
    ρ::Real,
    ρold::Real,
    ρbarold::Real,
    ζ::Real,
    ρbar::Real,
    θnew::Real
) where {T}
    δ = -θbar * ρ / (ρold * ρbarold)
    σ = ζ / (ρ * ρbar)
    ν = -θnew / ρ

    @inbounds @batch threadlocal=zero(T)::T for I in eachindex(x)
        h̄[I] = muladd(δ, h̄[I], h[I])
        x[I] = muladd(σ, h̄[I], x[I])
        h[I] = muladd(ν, h[I], v[I])
        threadlocal = muladd(conj(x[I]), x[I], threadlocal)
    end

    normx = sqrt(sum(real, threadlocal::Vector{T}))

    return h̄, x, h, normx
end

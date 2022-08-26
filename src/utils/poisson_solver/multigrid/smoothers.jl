#####
##### Sweeps
#####

abstract type Sweep end

struct BackwardSweep <: Sweep end
struct ForwardSweep <: Sweep end
struct SymmetricSweep <: Sweep end

struct BoundarySweep{S <: Sweep} <: Sweep end
BoundarySweep() = BoundarySweep{SymmetricSweep}()
BoundarySweep(S::Sweep) = BoundarySweep{typeof(S)}()


#####
##### Smoothers
#####

abstract type AbstractSmoother end


struct Smoothers{N, T} <: AbstractSmoother
    s::T
    function Smoothers(xs...)
        all(isa.(xs, AbstractSmoother)) || throw(ArgumentError(""))
        new{length(xs), typeof(xs)}(xs)
    end
end

function (s::Smoothers{N})(x, A, b) where {N}
    for i in 1:N
        x = s.s[i](x, A, b)
    end
    return x
end


struct EarlyStopping{N, S<:AbstractSmoother} <: AbstractSmoother
    smoother::S
    tol::Float64
end

EarlyStopping(smoother::AbstractSmoother, tol::Real, iter::Integer) =
    EarlyStopping{Int(iter), typeof(smoother)}(smoother, tol)

function (s::EarlyStopping{N})(x, A, b) where {N}
    ϵ = s.tol*norm(b)
    for _ in 1:N
        __norm_residual(b, A, x) ≤ ϵ && break
        s.smoother(x, A, b)
    end
    return x
end


#####
##### Gauss-Seidel
#####

struct GaussSeidel{S, I, RB} <: AbstractSmoother end

GaussSeidel(s::Sweep = SymmetricSweep(); iter::Integer = 1, rb::Bool = false) =
    GaussSeidel{typeof(s), Int(iter), rb}()

RedBlackGaussSeidel(s::Sweep = SymmetricSweep(); iter::Integer = 1) =
    GaussSeidel{typeof(s), Int(iter), true}()


function (s::GaussSeidel{S, I, false})(
    x::AbstractArray{T, N},
    A::Poisson2{T, N},
    b::AbstractArray{T, N},
) where {S<:Sweep, I, T, N}
    Rf = ntuple(n -> 2:size(x, n)-1, Val(N))
    Rb = ntuple(n -> reverse(Rf[n]), Val(N))

    for _ in 1:I
        if S === ForwardSweep || S === SymmetricSweep
            gs!(x, A, b, A.interior, Rf)
        end
        if S === BackwardSweep || S === SymmetricSweep
            gs!(x, A, b, A.interior, Rb)
        end
    end

    return x
end

function (s::GaussSeidel{BoundarySweep{S}, I, false})(
    x::AbstractArray{T, N},
    A::Poisson2{T, N},
    b::AbstractArray{T, N},
) where {S<:Sweep, I, T, N}
    Rf = ntuple(n -> 2:size(x, n)-1, Val(N))
    Rb = ntuple(n -> reverse(Rf[n]), Val(N))

    for _ in 1:I
        if S === ForwardSweep || S === SymmetricSweep
            gs!(x, A, b, A.boundary, Rf)
        end
        if S === BackwardSweep || S === SymmetricSweep
            gs!(x, A, b, A.boundary, Rb)
        end
    end

    return x
end


function (s::GaussSeidel{S, I, true})(
    x::AbstractArray{T, N},
    A::Poisson2{T, N},
    b::AbstractArray{T, N},
) where {S<:Sweep, I, T, N}
    Cf = 1:2
    Cb = 2:3
    Cs = 1:3

    for _ in 1:I
        if S === ForwardSweep
            rbgs!(x, A, b, A.interior, A.R8, Cf)
        end
        if S === BackwardSweep
            rbgs!(x, A, b, A.interior, A.R8, Cb)
        end
        if S === SymmetricSweep
            rbgs!(x, A, b, A.interior, A.R8, Cs)
        end
    end

    return x
end

function (s::GaussSeidel{BoundarySweep{S}, I, true})(
    x::AbstractArray{T, N},
    A::Poisson2{T, N},
    b::AbstractArray{T, N},
) where {S<:Sweep, I, T, N}
    Cf = 1:2
    Cb = 2:3

    for _ in 1:I
        if S === ForwardSweep || S === SymmetricSweep
            rbgs!(x, A, b, A.boundary, A.R8, Cf)
        end
        if S === BackwardSweep || S === SymmetricSweep
            rbgs!(x, A, b, A.boundary, A.R8, Cb)
        end
    end

    return x
end


function gs!(
    x::AbstractArray{T, 3},
    A::Poisson2{T, 3},
    b::AbstractArray{T, 3},
    m::AbstractArray{Bool, 3},
    R::NTuple{3},
) where {T}
    I, J, K = R
    iD = A.iD
    idx2 = -iD * A.idx2[1]
    idy2 = -iD * A.idx2[2]
    idz2 = -iD * A.idx2[3]

    @inbounds for k in K
        for j in J
            for i in I
                if m[i,j,k]
                    Δz = x[i,j,k-1] + x[i,j,k+1]
                    Δy = x[i,j-1,k] + x[i,j+1,k]

                    Δ = iD * b[i,j,k]
                    Δ = muladd(idz2, Δz, Δ)
                    Δ = muladd(idy2, Δy, Δ)
                    Δ = muladd(idx2, x[i+1,j,k], Δ)
                    Δ = muladd(idx2, x[i-1,j,k], Δ)

                    x[i,j,k] = Δ
                end
            end
        end
    end

    return x
end


function rbgs!(
    x::AbstractArray{T, 3},
    A::Poisson2{T, 3},
    b::AbstractArray{T, 3},
    m::AbstractArray{Bool, 3},
    R::Array{NTuple{3, UnitRange{Int}}},
    C::AbstractRange{Int},
) where {T}
    iD = A.iD
    idx2 = -iD * A.idx2[1]
    idy2 = -iD * A.idx2[2]
    idz2 = -iD * A.idx2[3]

    for s in C
        @batch minbatch=8 for (I, J, K) in R
            for k in K
                ks = Bool(k & 1)
                for j in J
                    js = Bool(j & 1)
                    ss = Bool((s + Int((ks && js) || (!ks && !js))) & 1)
                    for i in I
                        z1 = x[i,j,k-1]
                        y1 = x[i,j-1,k]
                        x1 = x[i-1,j,k]
                        x2 = x[i+1,j,k]
                        y2 = x[i,j+1,k]
                        z2 = x[i,j,k+1]

                        is = Bool(i & 1)
                        if ((!ss && !is) || (ss && is)) && m[i,j,k]
                            Δz = z1 + z2
                            Δy = y1 + y2
                            Δx = x1 + x2

                            Δ = iD * b[i,j,k]
                            Δ = muladd(idz2, Δz, Δ)
                            Δ = muladd(idy2, Δy, Δ)
                            Δ = muladd(idx2, Δx, Δ)

                            x[i,j,k] = Δ
                        end
                    end
                end
            end
        end
    end

    return x
end

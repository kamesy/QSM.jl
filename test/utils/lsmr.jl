#####
##### Tests copied from IterativeSolvers.jl
##### https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/v0.9.2/src/lsmr.jl
#####

# Type used in Dampenedtest
# solve (A'A + diag(v).^2 ) x = A'b
# using LSMR in the augmented space Ã = [A ; diag(v)] b̃ = [b; zeros(size(A, 2)]
struct DampenedMatrix{Tv,TA<:AbstractMatrix{Tv},TD<:AbstractVector{Tv}} <: AbstractMatrix{Tv}
    A::TA
    diagonal::TD
end

function Base.size(A::DampenedMatrix)
    m, n = size(A.A)
    l = length(A.diagonal)
    (m + l, n)
end

function Base.size(A::DampenedMatrix, dim::Integer)
    m, n = size(A.A)
    l = length(A.diagonal)
    dim == 1 ? (m + l) : (dim == 2 ? n : 1)
end

function LinearAlgebra.mul!(y::AbstractVector{Tv}, mw::DampenedMatrix, x::AbstractVector{Tv}) where {Tv}
    m₁ = size(mw.A, 1)
    m₂ = size(mw, 1)
    mul!(view(y, 1:m₁), mw.A, x)
    y[m₁+1:m₂] .= mw.diagonal .* x
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, mw::Adjoint{Tv,<:DampenedMatrix}, x::AbstractVector) where {Tv}
    m₁ = size(mw.parent.A, 1)
    m₂ = size(mw.parent, 1)
    mul!(y, adjoint(mw.parent.A), view(x, 1:m₁))
    y .+= mw.parent.diagonal .* view(x, m₁+1:m₂)
    return y
end

"""
Produces the m × n submatrix from
A = [ 1
      1 2
        2 3
          3 4
            ...
              n ]
suitably padded by zeros.
"""
function sol_matrix(m, n)
    mn = min(m, n)
    I, J, V = SparseArrays.spdiagm_internal(-1 => 1.0 : mn - 1, 0 => 1.0 : mn)
    sparse(I, J, V, m, n)
end

@testset "Small dense matrix" for T = (Float32, Float64, ComplexF32, ComplexF64)
    Random.seed!(501)
    A = rand(T, 10, 5)
    b = rand(T, 10)
    x = QSM.lsmr(A, b)
    @test norm(x - A\b) ≤ √eps(real(T))
end

@testset "SOL test" for (m, n, damp) = ((10, 10, 0), (20, 10, 0), (20, 10, 0.1))
    # Test adapted from the BSD-licensed Matlab implementation at
    #    http://www.stanford.edu/group/SOL/software/lsqr.html
    #              Michael Saunders, Systems Optimization Laboratory,
    #              Dept of MS&E, Stanford University.
    #-----------------------------------------------------------------------
    # 11 Apr 1996: First version for distribution with lsqr.m.
    #              Michael Saunders, Dept of EESOR, Stanford University.

    A = sol_matrix(m, n)
    x = float(n : -1 : 1)
    b = A * x
    x_lsmr = QSM.lsmr(A, b, atol = 1e-7, btol = 1e-7, conlim = 1e10, maxit = 10n)
    @test norm(b - A * x) ≤ 1e-4
end

@testset "Dampened test" for (m, n) = ((10, 10), (20, 10))
    Random.seed!(501)
    # Test used to make sure A, b can be generic matrix / vector
    b = rand(m)
    A = rand(m, n)
    v = rand(n)
    A′ = DampenedMatrix(A, v)
    b′ = [b; zeros(n)]
    x = QSM.lsmr(A′, b′)
    @test norm((A'A + Matrix(Diagonal(v)) .^ 2)x - A'b) ≤ 1e-3
end

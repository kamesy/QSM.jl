using QSM: Poisson2
using QSM: wrapped_laplacian
using QSM: wrapped_laplacian_boundary_neumann!
using QSM: wrapped_laplacian_boundary_periodic!
using QSM: solve_poisson_dct, solve_poisson_fft, solve_poisson_mgpcg


L2norm(x) = norm(x) / sqrt(length(x))

function test_problem(::Type{T}, sz::NTuple{3, Integer}) where {T<:AbstractFloat}
    f = sinpi

    R = CartesianIndices(map(n -> 2:n-1, sz))
    m = zeros(Bool, sz)
    u = zeros(T, sz)
    d2u = zeros(T, sz)
    d2uh = zeros(T, sz)

    X = range(0, 1, length=sz[1])
    Y = range(0, 1, length=sz[2])
    Z = range(0, 1, length=sz[3])

    dx = (X.step.hi, Y.step.hi, Z.step.hi)
    _u = [f(x).*f(y).*f(z) for x in X, y in Y, z in Z]

    m[R] .= true
    u[R] .= _u[R]
    d2u[R] .= -3*π^2*_u[R]

    A = Poisson2(T, m, dx)
    d2uh = mul!(d2uh, A, u)

    return A, u, d2u, d2uh
end


@testset "Multigrid" begin
    include("multigrid/multigrid.jl")
end

@testset "DCT with wrapped Laplacian" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(11, 12, 13), (12, 13, 14, 3)]
        x, dx = rand(T, sz), Tuple(rand(T, 3))
        x .-= mean(x, dims=(1,2,3))
        d2x = wrapped_laplacian(x, dx)
        d2x = wrapped_laplacian_boundary_neumann!(d2x, x, dx)
        @test solve_poisson_dct(d2x, dx) ≈ x
    end
end

@testset "FFT with wrapped Laplacian" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(11, 12, 13), (12, 13, 14, 3)]
        x, dx = rand(T, sz), Tuple(rand(T, 3))
        x .-= mean(x, dims=(1,2,3))
        d2x = wrapped_laplacian(x, dx)
        d2x = wrapped_laplacian_boundary_periodic!(d2x, x, dx)
        @test solve_poisson_fft(d2x, dx) ≈ x
    end

end

@testset "MGPCG with wrapped_laplacian" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(127, 128, 129), (130, 131, 132)]
        A, u, d2u, d2uh = test_problem(T, sz)
        @test wrapped_laplacian(u, A.dx) ≈ d2uh
        @test solve_poisson_mgpcg(d2uh, A.interior, A.dx; rtol=eps(T)) ≈ u
    end
end

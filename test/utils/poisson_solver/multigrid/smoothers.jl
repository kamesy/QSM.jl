using QSM: GaussSeidel
using QSM: BackwardSweep, ForwardSweep, SymmetricSweep
using QSM: Poisson2, __norm_residual, residual!


@testset "Gauss-Seidel" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(16, 17, 18), (19, 20, 21)]
        maxit = 2*prod(sz)
        R = CartesianIndices(map(n -> 2:n-1, sz))

        m = zeros(Bool, sz)
        b = zeros(T, sz)
        x0 = zeros(T, sz)

        m[R] .= true
        b[R] .= randn(size(R))
        x0[R] .= randn(size(R))

        A = Poisson2(T, m, Tuple(rand(T, 3)))
        x1 = mul!(copy(x0), A, x0)

        for S in (SymmetricSweep, ForwardSweep, BackwardSweep)
            for RB in (false, true)
                solver = GaussSeidel(S(); iter=maxit, rb=RB)

                y = solver(copy(x0), A, b)
                @test __norm_residual(b, A, y) / norm(b) ≤ sqrt(eps(T))
                @test norm(residual!(copy(y), b, A, y)) / norm(b) ≤ sqrt(eps(T))

                # Fixed-point test.
                # Relaxation should not alter the exact solution.
                # Use exact solution as initial guess.
                solver = GaussSeidel{S, 1, RB}()
                @test solver(copy(x0), A, x1) ≈ x0

                solver = GaussSeidel{S, 25, RB}()
                @test solver(copy(x0), A, x1) ≈ x0
            end
        end
    end
end

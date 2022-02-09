using QSM: V, W, F, Multigrid, cycle!
using QSM: BackwardSweep, ForwardSweep, SymmetricSweep, EarlyStopping, GaussSeidel
using QSM: restrict!, correct_prolong!
using QSM: residual!, tfill!


#####
##### TODO: Smoothing factor check
#####

@testset "Smoothers" begin
    include("smoothers.jl")
end

@testset "Transfer" begin
    include("transfer.jl")
end

@testset "Fixed point" for T in (Float32, Float64)
    # Cycling should not alter the exact solution.
    # Use exact solution as initial guess.
    for sz in [(128, 129, 130), (131, 132, 133)]
        A, u, d2u, d2uh = test_problem(T, sz)

        M = Multigrid(T, A,
            coarsesolver = GaussSeidel(SymmetricSweep(), iter=2500),
            presmoother = GaussSeidel(SymmetricSweep(), iter=1),
            postsmoother = GaussSeidel(SymmetricSweep(), iter=1),
        )

        for cycle in (V(), W(), F())
            x = M.workspace.x[1] = copy(u)
            M.workspace.b[1] = copy(d2uh)
            cycle!(M, cycle)
            @test x ≈ u
        end
    end
end

@testset "Two-level Cycle" for T in (Float32, Float64)
    # Test intergrid operators.
    # Postsmooth residuals should be lower than presmooth residuals.
    for sz in [(32, 33, 34), (35, 36, 37)]
        A, u, d2u, d2uh = test_problem(T, sz)

        M = Multigrid(T, A,
            coarsesolver = EarlyStopping(
                GaussSeidel(SymmetricSweep(), iter=10),
                sqrt(eps(T)),
                1_000_000
            ),
            presmoother = GaussSeidel(ForwardSweep(), iter=2),
            postsmoother = GaussSeidel(BackwardSweep(), iter=2),
            maxlevel = 2
        )

        nd0 = Inf
        nu0 = Inf
        M.workspace.x[1] = zeros(T, sz)
        M.workspace.b[1] = copy(d2uh)

        l = 1
        for i in 1:5
            A = M.grids[l].A
            Ac = M.grids[l+1].A

            presmooth! = M.grids[l].presmoother
            postsmooth! = M.grids[l].postsmoother

            x = M.workspace.x[l]
            b = M.workspace.b[l]
            r = M.workspace.r[l]

            xc = M.workspace.x[l+1]
            bc = M.workspace.b[l+1]

            presmooth!(x, A, b)
            residual!(r, b, A, x)
            restrict!(bc, Ac, r)

            xc = tfill!(xc, 0, Ac)
            M.coarsesolver(xc, Ac, bc)

            correct_prolong!(x, A, xc)
            postsmooth!(x, A, b)

            nd = L2norm(r)
            nu = L2norm(residual!(r, b, A, x))
            @test nd < nd0
            @test nu < nu0
            @test nu < nd
            nd0, nu0 = nd, nu
        end
    end
end

using QSM: prolong!, restrict!, restrict_size
using QSM: Poisson2


@testset "Restriction" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(16, 17, 18), (19, 20, 21)]
        szc = restrict_size(sz)

        R = CartesianIndices(map(n -> 2:n-1, sz))
        Rc = CartesianIndices(map(n -> 2:n-1, szc))

        x = zeros(T, sz)
        xc = zeros(T, szc)
        yc = zeros(T, szc)
        mxc = zeros(Bool, szc)
        myc = zeros(Bool, szc)

        x[R] .= 1
        xc[Rc] .= randn(size(Rc))
        yc[Rc] .= randn(size(Rc))

        mxc[Rc] .= true
        myc[Rc] .= rand(Bool, size(Rc))

        dx = Tuple(rand(T, 3))
        Axc = Poisson2(T, mxc, dx)
        Ayc = Poisson2(T, myc, dx)

        restrict!(xc, Axc, x)
        restrict!(yc, Ayc, x)

        @test xc[Rc] ≈ ones(size(Rc))
        @test sum(xc[Rc]) ≈ sum(xc)
        @test myc.*yc ≈ myc.*xc
    end
end

@testset "Prolongation" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(16, 17, 18), (19, 20, 21)]
        szc = restrict_size(sz)

        R = CartesianIndices(map(n -> 3:n-2, sz))
        Rc = CartesianIndices(map(n -> 2:n-1, szc))

        xc = zeros(T, szc)
        x = zeros(T, sz)
        y = zeros(T, sz)
        mx = zeros(Bool, sz)
        my = zeros(Bool, sz)

        xc .= 1
        x[R] .= randn(size(R))
        y[R] .= randn(size(R))

        mx[R] .= true
        my[R] .= rand(Bool, size(R))

        dx = Tuple(rand(T, 3))
        Ax = Poisson2(T, mx, dx)
        Ay = Poisson2(T, my, dx)

        prolong!(x, Ax, xc)
        prolong!(y, Ay, xc)

        @test x[R] ≈ ones(size(R))
        @test my.*y ≈ my.*x

        R = CartesianIndices(map(n -> 2:n-1, sz))
        @test sum(x[R]) ≈ sum(x)
    end
end

@testset "R = αP^T" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(16, 17, 18), (19, 20, 21)]
        szc = restrict_size(sz)

        u = zeros(T, sz)
        v = zeros(T, sz)
        m = zeros(Bool, sz)

        uc = zeros(T, szc)
        vc = zeros(T, szc)
        mc = zeros(Bool, szc)

        R = CartesianIndices(map(n -> 2:n-1, sz))
        Rc = CartesianIndices(map(n -> 2:n-1, szc))

        m[R] .= true
        v[R] .= randn(size(R))
        u[R] .= randn(size(R))

        mc[Rc] .= true
        uc[Rc] .= randn(size(Rc))
        vc[Rc] .= randn(size(Rc))

        dx = Tuple(rand(T, 3))
        A = Poisson2(T, m, dx)

        dxc = ((sz.-1)./(szc.-1)) .* dx
        Ac = Poisson2(T, mc, dxc)

        prolong!(u, A, uc)
        restrict!(vc, Ac, v)

        R = CartesianIndices(map(n -> 1:(iseven(n) ? n-2 : n), sz))
        @test dot(uc, vc) ≈ dot(v[R], inv(8).*u[R])
    end
end

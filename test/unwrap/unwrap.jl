using QSM: lap, wrapped_laplacian


@testset "Wrapped Laplacian" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(11,12,13), (12,13,14,5)]
        x, dx = rand(T, sz), Tuple(rand(T, 3))
        @test wrapped_laplacian(x, dx) ≈ lap(x, dx)
    end
end

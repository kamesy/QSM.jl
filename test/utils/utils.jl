using QSM: padfastfft, padarray!, unpadarray!


@testset "LSMR" begin
    include("lsmr.jl")
end

@testset "Poisson solver" begin
    include("poisson_solver/poisson_solver.jl")
end

@testset "Multi-echo" begin
    include("multi_echo.jl")
end

@testset "padfastfft and psf2otf" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(8, 9, 10), (11, 12, 13)]
        for ksz in [(3, 4, 3), (4, 3, 4)]
            x = randn(T, sz)
            k = randn(T, ksz)
            y = zeros(T, sz)
            z = zeros(T, sz)

            y = unpadarray!(y, conv(k, x))

            for _rfft in (false, true)
                xp = padfastfft(x, ksz, rfft=_rfft)
                K = psf2otf(k, size(xp), rfft=_rfft)

                if _rfft
                    @test iseven(size(xp, 1))
                    P = plan_rfft(xp)
                    zp = P\(K.*(P*xp))
                else
                    P = plan_fft(xp)
                    zp = real(P\(K.*(P*xp)))
                end

                @test unpadarray!(z, zp) ≈ y
            end
        end
    end
end

@testset "padarray!" begin
    A = reshape(1:25, 5, 5)

    @test padarray!(rand(Int, (8, 8)), A, :fill, 0) ==
        [
            0  0  0   0   0   0   0  0
            0  0  0   0   0   0   0  0
            0  0  1   6  11  16  21  0
            0  0  2   7  12  17  22  0
            0  0  3   8  13  18  23  0
            0  0  4   9  14  19  24  0
            0  0  5  10  15  20  25  0
            0  0  0   0   0   0   0  0
        ]

    @test padarray!(rand(Int, (8, 9)), A, :circular) ==
        [
            19  24  4   9  14  19  24  4   9
            20  25  5  10  15  20  25  5  10
            16  21  1   6  11  16  21  1   6
            17  22  2   7  12  17  22  2   7
            18  23  3   8  13  18  23  3   8
            19  24  4   9  14  19  24  4   9
            20  25  5  10  15  20  25  5  10
            16  21  1   6  11  16  21  1   6
        ]

    @test padarray!(rand(Int, (9, 8)), A, :replicate) ==
        [
            1  1  1   6  11  16  21  21
            1  1  1   6  11  16  21  21
            1  1  1   6  11  16  21  21
            2  2  2   7  12  17  22  22
            3  3  3   8  13  18  23  23
            4  4  4   9  14  19  24  24
            5  5  5  10  15  20  25  25
            5  5  5  10  15  20  25  25
            5  5  5  10  15  20  25  25
        ]

    @test padarray!(rand(Int, (9, 9)), A, :symmetric) ==
        [
             7  2  2   7  12  17  22  22  17
             6  1  1   6  11  16  21  21  16
             6  1  1   6  11  16  21  21  16
             7  2  2   7  12  17  22  22  17
             8  3  3   8  13  18  23  23  18
             9  4  4   9  14  19  24  24  19
            10  5  5  10  15  20  25  25  20
            10  5  5  10  15  20  25  25  20
             9  4  4   9  14  19  24  24  19
        ]

    @test padarray!(rand(Int, (9, 9)), A, :reflect) ==
        [
            13   8  3   8  13  18  23  18  13
            12   7  2   7  12  17  22  17  12
            11   6  1   6  11  16  21  16  11
            12   7  2   7  12  17  22  17  12
            13   8  3   8  13  18  23  18  13
            14   9  4   9  14  19  24  19  14
            15  10  5  10  15  20  25  20  15
            14   9  4   9  14  19  24  19  14
            13   8  3   8  13  18  23  18  13
        ]
end

@testset "Average" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(8, 9, 10), (11, 12, 13)]
        for NT in (1, 2, 5, 6, 12, 32, 48, 96)
            y = zeros(T, sz)
            x = randn(T, sz..., NT)
            t = rand(T, NT)
            W = rand(T, sz..., NT)
            m = rand(Bool, sz...)

            z = sum(x, dims=4) / NT
            y = multi_echo_average!(y, x, nothing, nothing, nothing)
            @test isapprox(y, z, atol=100*eps(T))

            y .*= m.*rand(T)
            y = multi_echo_average!(y, x, nothing, nothing, m)
            @test isapprox(y, m.*z, atol=100*eps(T))

            y .*= rand(T)
            w = reshape(t, 1, 1, 1, NT)
            w ./= sum(w)
            z = sum(w .* x, dims=4)
            y = multi_echo_average!(y, x, t, nothing, nothing)
            @test isapprox(y, z, atol=100*eps(T))

            y .*= m.*rand(T)
            y = multi_echo_average!(y, x, t, nothing, m)
            @test isapprox(y, m.*z, atol=100*eps(T))

            y .*= rand(T)
            w = W ./ sum(W, dims=4)
            z = sum(w .* x, dims=4)
            y = multi_echo_average!(y, x, nothing, W, nothing)
            @test isapprox(y, z, atol=100*eps(T))

            y .*= m.*rand(T)
            y = multi_echo_average!(y, x, nothing, W, m)
            @test isapprox(y, m.*z, atol=100*eps(T))

            y .*= rand(T)
            w = reshape(t, 1, 1, 1, NT) .* W
            w ./= sum(w, dims=4)
            z = sum(w .* x, dims=4)
            y = multi_echo_average!(y, x, t, W, nothing)
            @test isapprox(y, z, atol=100*eps(T))

            y .*= m.*rand(T)
            y = multi_echo_average!(y, x, t, W, m)
            @test isapprox(y, m.*z, atol=100*eps(T))
        end
    end
end

@testset "Linear fit, no intercept" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(8, 9, 10), (11, 12, 13)]
        for NT in (1, 2, 5, 6, 12, 32, 48, 96)
            β = randn(T, sz)
            b = randn(T, sz)
            x = rand(T, NT)
            W = fill(rand(T), sz..., NT)
            m = rand(Bool, sz...)

            y = reshape(x, 1, 1, 1, NT) .* β

            b = multi_echo_linear_fit!(b, y, x, nothing, nothing)
            @test isapprox(b, β, atol=250*eps(T))

            b .*= m.*rand(T)
            b = multi_echo_linear_fit!(b, y, x, nothing, m)
            @test isapprox(b, m.*β, atol=250*eps(T))

            b .*= rand(T)
            b = multi_echo_linear_fit!(b, y, x, W, nothing)
            @test isapprox(b, β, atol=250*eps(T))

            b .*= m.*rand(T)
            b = multi_echo_linear_fit!(b, y, x, W, m)
            @test isapprox(b, m.*β, atol=250*eps(T))
        end
    end
end

@testset "Linear fit, yes intercept" for T in (Float32, Float64)
    Random.seed!(501)
    for sz in [(8, 9, 10), (11, 12, 13)]
        for NT in (2, 5, 6, 12, 32, 48, 96)
            β = randn(T, sz)
            α = randn(T, sz)
            b = randn(T, sz)
            a = randn(T, sz)
            x = rand(T, NT)
            W = fill(rand(T), sz..., NT)
            m = rand(Bool, sz...)

            y = reshape(x, 1, 1, 1, NT) .* β .+ α

            b, a = multi_echo_linear_fit!(b, a, y, x, nothing, nothing)
            @test isapprox(b, β, atol=1000*eps(T))
            @test isapprox(a, α, atol=1000*eps(T))

            b .*= m.*rand(T)
            a .*= m.*rand(T)
            b, a = multi_echo_linear_fit!(b, a, y, x, nothing, m)
            @test isapprox(b, m.*β, atol=1000*eps(T))
            @test isapprox(a, m.*α, atol=1000*eps(T))

            b .*= rand(T)
            a .*= rand(T)
            b, a = multi_echo_linear_fit!(b, a, y, x, W, nothing)
            @test isapprox(b, β, atol=1000*eps(T))
            @test isapprox(a, α, atol=1000*eps(T))

            b .*= m.*rand(T)
            a .*= m.*rand(T)
            b, a = multi_echo_linear_fit!(b, a, y, x, W, m)
            @test isapprox(b, m.*β, atol=1000*eps(T))
            @test isapprox(a, m.*α, atol=1000*eps(T))
        end
    end
end

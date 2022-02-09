using Aqua
using DSP: conv
using FFTW
using LinearAlgebra
using QSM
using Random
using SparseArrays
using Statistics: mean
using Test


@testset "Unwrap" begin
    include("unwrap/unwrap.jl")
end

@testset "Utils" begin
    include("utils/utils.jl")
end

#@testset "QSM.jl" begin
#    Aqua.test_all(QSM)
#end

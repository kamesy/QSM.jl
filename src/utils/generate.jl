using FFTW
using MAT
using Random
using Printf
using StaticArrays

include("models.jl")


"""
Generate magnetic susceptibility maps and the corresponding fields due to
randomly placed spheres and randomly oriented infinite cylinders subject to a
static B-field inside a 3d rectangular volume.

Method
------
    1. Pick array size, voxel size for grid (N/n grid)
    2. for n iterations
        i.   pick number of tries, ntry, to check for valid config
        ii.  randomly place sources inside grid until no valid config can be
             found ntry times in a row
        iii. valid config means no overlapping structures. concentric allowed.
"""
function generate(
    sz::NTuple{N, Integer},
    vsz::NTuple{3, Real};
    outpath::Union{Nothing, AbstractString} = nothing,
    nsources::Integer = typemax(Int)-1,
    density::Real = 0.51,
    bdir::NTuple{3, Real} = (0, 0, 1),
    χμ::Float64 = 1e-6,
    χσ::Float64 = 0.07,
    seed::Int = 12345,
) where {N}
    @assert nsources > 0
    @assert density > 0 && density < 1
    Random.seed!(seed)

    if !(norm(bdir) ≈ 1)
        @warn "B direction vector is not normalized. Normalizing..."
        bdir = bdir ./ norm(bdir)
    end

    x = zeros(sz)
    f = zeros(sz)
    m = Array{Bool}(falses(sz))

    rmax = cld(minimum(sz), 5)
    rmax = 7
    Cs = ntuple(i -> rmax+1:sz[i]-(rmax+1)-1, Val(3))
    Cs = ntuple(i -> 10*rmax+1:sz[i]-(10*rmax+1)-1, Val(3))

    ns = 0
    while true
        model = rand((:sphere, :cylinder))
        model = :sphere

        r = rand(3:rmax)
        c = rand.(Cs)
        χ = χμ + χσ*randn()

        if model == :sphere
            x, f, m = sphere!(x, f, m, c, r, χ, bdir, vsz)
            ns += 1

        elseif model == :cylinder
            θ = π*rand()
            ϕ = 2π*rand()

            x, f, m = cylinder!(x, f, m, c, r, θ, ϕ, χ, bdir, vsz)
            ns += 1
        end

        if sum(m)/length(m) > density || ns >= nsources
            break
        end
    end

    if outpath !== nothing
        matwrite(outpath, Dict(:x => x, :f => f, :m => m), compress=true)
    end

    return x, f, m
end

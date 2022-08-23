module QSM


using Base.Threads: nthreads
using CPUSummary: num_cores
using FastPow: @fastpow
using IrrationalConstants: inv2π, twoπ
using LinearMaps: LinearMap
using NIfTI: NIVolume, niread, niwrite
using Polyester: @batch
using PolyesterWeave: reset_workers!
using Printf: @printf
using SLEEFPirates: pow, sincos_fast
using Static: known
using StaticArrays: SVector
using ThreadingUtilities: initialize_task
using TiledIteration: EdgeIterator, TileIterator, padded_tilesize

using LinearAlgebra
using FFTW


export bet
export gradfp, gradfp!, gradfp_adj, gradfp_adj!, lap, lap!
export dipole_kernel, laplace_kernel, smv_kernel
export fit_echo_linear, fit_echo_linear!
export r2star_arlo, r2star_crsi, r2star_ll, r2star_numart2s
export crop_mask, crop_indices, erode_mask, erode_mask!
export fastfftsize, padfastfft, padarray!, unpadarray, unpadarray!, psf2otf
include("utils/utils.jl")

export unwrap_laplacian
include("unwrap/unwrap.jl")

export ismv, lbv, pdf, sharp, vsharp
include("bgremove/bgremove.jl")

export nltv, rts, tikh, tkd, tsvd, tv
include("inversion/inversion.jl")


function __init__()
    @static if FFTW.fftw_provider == "fftw"
        fftw_set_threading(:FFTW)
    end
    return nothing
end


#####
##### Polyester.jl
#####
function reset_threading()
    # after user interrupt during @batch loop, threading has to be reset:
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    reset_workers!()
    foreach(initialize_task, 1:min(nthreads(), (Sys.CPU_THREADS)::Int) - 1)
    return nothing
end


#####
##### FFTW.jl
#####
const FFTW_NTHREADS = Ref{Int}(known(num_cores()))


@static if FFTW.fftw_provider == "fftw"
    # modified `FFTW.spawnloop` to use Polyester for multi-threading
    # https://github.com/JuliaMath/FFTW.jl/blob/v1.4.5/src/providers.jl#L49
    function _fftw_spawnloop(f::Ptr{Cvoid}, fdata::Ptr{Cvoid}, elsize::Csize_t, num::Cint, ::Ptr{Cvoid})
        @batch for i in 0:num-1
            ccall(f, Ptr{Cvoid}, (Ptr{Cvoid},), fdata + elsize*i)
        end
        return nothing
    end

    function fftw_set_threading(lib::Symbol = :FFTW)
        lib ∈ (:FFTW, :Polyester, :Threads) ||
            throw(ArgumentError("lib must be one of :FFTW, :Polyester or :Threads, got :$(lib)"))

        if lib ∈ (:Polyester, :Threads) && nthreads() < 2
            @warn "Cannot use $lib with FFTW. Defaulting to FFTW multi-threading" Threads.nthreads()
            lib = :FFTW
        end

        if lib == :Polyester
            cspawnloop = @cfunction(
                _fftw_spawnloop,
                Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid})
            )
        elseif lib == :Threads
            cspawnloop = @cfunction(
                FFTW.spawnloop,
                Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid})
            )
        else
            cspawnloop = C_NULL
        end

        ccall(
            (:fftw_threads_set_callback,  FFTW.libfftw3[]),
            Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cspawnloop, C_NULL
        )

        ccall(
            (:fftwf_threads_set_callback, FFTW.libfftw3f[]),
            Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cspawnloop, C_NULL
        )

        return nothing
    end
end


end # module

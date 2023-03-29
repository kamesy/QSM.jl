# QSM.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kamesy.github.io/QSM.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kamesy.github.io/QSM.jl/dev)
[![Build Status](https://github.com/kamesy/QSM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kamesy/QSM.jl/actions/workflows/CI.yml?query=branch%3Amain)

Julia toolbox for quantitative susceptibility mapping (QSM).

## Installation

QSM.jl requires Julia v1.6 or later.

```julia
julia> ]add QSM
```

## Example

```julia
using QSM

# constants
γ = 267.52      # gyromagnetic ratio
B0 = 3          # main magnetic field strength

# load 3D single-, or multi-echo data using your favourite
# package, e.g. MAT.jl, NIfTI.jl, ParXRec.jl, ...
mag, phas = ...

bdir = (...,)   # direction of B-field
vsz  = (...,)   # voxel size
TEs  = [...]    # echo times

# extract brain mask from last echo using FSL's bet
mask0 = bet(@view(mag[:,:,:,end]), vsz, "-m -n -f 0.5")

# erode mask
mask1 = erode_mask(mask0, 5)

# unwrap phase + harmonic background field correction
uphas = unwrap_laplacian(phas, mask1, vsz)

# convert units
@views for t in axes(uphas, 4)
    uphas[:,:,:,t] .*= inv(B0 * γ * TEs[t])
end

# remove non-harmonic background fields
fl, mask2 = vsharp(uphas, mask1, vsz)

# dipole inversion
x = rts(fl, mask2, vsz, bdir=bdir)
```

## Multi-Threading
Multi-threading is provided by [`Polyester.jl`](https://github.com/JuliaSIMD/Polyester.jl). To enable threading, [`start Julia with multiple threads`](https://docs.julialang.org/en/v1.6/manual/multi-threading/#Starting-Julia-with-multiple-threads):

```bash
julia --threads N
```
or
```bash
export JULIA_NUM_THREADS=N
```

After an interrupt of a multi-threaded loop, reset threading via:
```julia
julia> QSM.reset_threading()
```

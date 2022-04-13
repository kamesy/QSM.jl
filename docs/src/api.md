```@meta
CurrentModule = QSM
```

# API Reference

```@contents
Pages = ["api.md"]
```

## Index
```@index
```

## Phase unwrapping
```@docs
unwrap_laplacian
```

## Background field correction
```@docs
ismv
lbv
pdf
sharp
vsharp
```

## Dipole inversion
```@docs
nltv
rts
tikh
tkd
tsvd
tv
```

## Binary masks
```@docs
bet
crop_indices
crop_mask
erode_mask
erode_mask!
```

## Kernels
```@docs
dipole_kernel
laplace_kernel
smv_kernel
```

## Multi-echo
```@docs
fit_echo_linear
fit_echo_linear!
```

## Other
### R2* mapping
```@docs
r2star_arlo
r2star_crsi
r2star_ll
r2star_numart2s
```

### Finite differences
```@docs
gradfp
gradfp!
gradfp_adj
gradfp_adj!
lap
lap!
```

### Padding
```@docs
fastfftsize
padarray!
padfastfft
unpadarray
unpadarray!
```

### Miscellaneous
```@docs
psf2otf
```

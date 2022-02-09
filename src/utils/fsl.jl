"""
    bet(
        mag::AbstractArray{<:Real, 3},
        vsz::NTuple{3, Real},
        betargs::AbstractString = "-m -n -f 0.5"
    ) -> Array{Bool, 3}

Interface to FSL's bet.

### Arguments
- `mag::AbstractArray{<:Real, 3}`: magnitude image
- `vsz::NTuple{3, Real}`: voxel size
- `betargs::AbstractString = "-m -n -f 0.5"`: bet options

### Returns
- `Array{Bool, 3}`: binary brain mask
"""
function bet(
    mag::AbstractArray{<:Real, 3},
    vsz::NTuple{3, Real},
    betargs::AbstractString = "-m -n -f 0.5"
)
    if Sys.which("bet") === nothing
        error("bet: command not found")
    end

    magfile = tempname(cleanup=false) * ".nii"
    maskfile = replace(magfile, ".nii" => "_mask.nii.gz")

    nii = NIVolume(mag, voxel_size=vsz)
    niwrite(magfile, nii)

    try
        # bet appends '_mask' to output filename regardless of user choice
        run(`bet $magfile $magfile $(split(betargs))`)
        nii = niread(maskfile)
    finally
        rm(magfile, force=true)
        rm(maskfile, force=true)
    end

    return Array{Bool, 3}(nii.raw)
end

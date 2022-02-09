function restrict(A::Poisson2{T}) where {T}
    interiorc = restrict(A.interior)
    sz = size(A.interior)
    szc = size(interiorc)
    dxc = ((sz .- 1) ./ (szc .- 1)) .* A.dx
    return Poisson2(T, interiorc, dxc)
end


function restrict(interior::AbstractArray{Bool, 3})
    sz = size(interior)
    szc = restrict_size(sz)
    return restrict!(tzero(interior, szc), interior)
end

function restrict!(mc::AbstractArray{Bool, 3}, m::AbstractArray{Bool, 3})
    nxc, nyc, nzc = size(mc)

    @inbounds @batch minbatch=8 for kc in 2:nzc-1
        k = (kc << 1) - 1
        for jc in 2:nyc-1
            j = (jc << 1) - 1
            for ic in 2:nxc-1
                i = (ic << 1) - 1

                mc[ic,jc,kc] =
                    m[i,j,k] &&
                    m[i+1,j,k] &&
                    m[i,j+1,k] &&
                    m[i+1,j+1,k] &&
                    m[i,j,k+1]   &&
                    m[i+1,j,k+1] &&
                    m[i,j+1,k+1] &&
                    m[i+1,j+1,k+1]
            end
        end
    end

    return mc
end


function restrict!(
    xc::AbstractArray{<:AbstractFloat, 3},
    Ac::Poisson2{<:AbstractFloat, 3},
    x::AbstractArray{T, 3},
) where {T<:AbstractFloat}
    a0 = convert(T, 8//64)
    a1 = convert(T, 4//64)
    a2 = convert(T, 2//64)
    a3 = convert(T, 1//64)

    @inbounds @batch minbatch=8 for (Ic, Jc, Kc) in Ac.R27
        for kc in Kc
            k = (kc << 1) - 1
            for jc in Jc
                j = (jc << 1) - 1
                for ic in Ic
                    i = (ic << 1) - 1

                    x000 = x[i-1,j-1,k-1]
                    x100 = x[i,j-1,k-1]
                    x200 = x[i+1,j-1,k-1]

                    x010 = x[i-1,j,k-1]
                    x110 = x[i,j,k-1]
                    x210 = x[i+1,j,k-1]

                    x020 = x[i-1,j+1,k-1]
                    x120 = x[i,j+1,k-1]
                    x220 = x[i+1,j+1,k-1]

                    x001 = x[i-1,j-1,k]
                    x101 = x[i,j-1,k]
                    x201 = x[i+1,j-1,k]

                    x011 = x[i-1,j,k]
                    x111 = x[i,j,k]
                    x211 = x[i+1,j,k]

                    x021 = x[i-1,j+1,k]
                    x121 = x[i,j+1,k]
                    x221 = x[i+1,j+1,k]

                    x002 = x[i-1,j-1,k+1]
                    x102 = x[i,j-1,k+1]
                    x202 = x[i+1,j-1,k+1]

                    x012 = x[i-1,j,k+1]
                    x112 = x[i,j,k+1]
                    x212 = x[i+1,j,k+1]

                    x022 = x[i-1,j+1,k+1]
                    x122 = x[i,j+1,k+1]
                    x222 = x[i+1,j+1,k+1]

                    if Ac.interior[ic,jc,kc]
                        x3 = x000 + x200 + x020 + x220 +
                             x002 + x202 + x022 + x222

                        x2 = x100 + x010 + x210 + x120 +
                             x001 + x201 + x021 + x221 +
                             x102 + x012 + x212 + x122

                        x1 = x110 + x101 + x011 + x211 +
                             x121 + x112

                        c = a0 * x111
                        c = muladd(a3, x3, c)
                        c = muladd(a2, x2, c)
                        c = muladd(a1, x1, c)

                        xc[ic,jc,kc] = c
                    end
                end
            end
        end
    end

    return xc
end


function prolong!(
    x::AbstractArray{<:AbstractFloat, 3},
    A::Poisson2{<:AbstractFloat, 3},
    xc::AbstractArray{T, 3},
) where {T<:AbstractFloat}
    a1 = convert(T, 4//8)
    a2 = convert(T, 2//8)
    a3 = convert(T, 1//8)

    nxc, nyc, nzc = size(xc)
    m = A.interior

    @inbounds @batch for kc in 1:nzc-1
        k = (kc << 1) - 1
        for jc in 1:nyc-1
            j = (jc << 1) - 1
            for ic in 1:nxc-1
                i = (ic << 1) - 1

                x000 = xc[ic,jc,kc]
                x100 = xc[ic+1,jc,kc]
                x010 = xc[ic,jc+1,kc]
                x110 = xc[ic+1,jc+1,kc]
                x001 = xc[ic,jc,kc+1]
                x101 = xc[ic+1,jc,kc+1]
                x011 = xc[ic,jc+1,kc+1]
                x111 = xc[ic+1,jc+1,kc+1]

                x1 = x000 + x100
                x2 = x000 + x010
                x3 = x000 + x001

                x4 = x010 + x110
                x5 = x001 + x101
                x6 = x001 + x011

                if m[i,j,k]
                    x[i,j,k] = x000
                end

                if m[i+1,j,k]
                    x[i+1,j,k] = a1*x1
                end

                if m[i,j+1,k]
                    x[i,j+1,k] = a1*x2
                end

                if m[i+1,j+1,k]
                    x[i+1,j+1,k] = a2*(x1 + x4)
                end

                if m[i,j,k+1]
                    x[i,j,k+1] = a1*x3
                end

                if m[i+1,j,k+1]
                    x[i+1,j,k+1] = a2*(x1 + x5)
                end

                if m[i,j+1,k+1]
                    x[i,j+1,k+1] = a2*(x2 + x6)
                end

                if m[i+1,j+1,k+1]
                    x[i+1,j+1,k+1] = a3*(x1 + x4 + x5 + x011 + x111)
                end
            end
        end
    end

    return x
end


function correct_prolong!(
    x::AbstractArray{<:AbstractFloat, 3},
    A::Poisson2{<:AbstractFloat, 3},
    xc::AbstractArray{T, 3},
) where {T}
    a1 = convert(T, 4//8)
    a2 = convert(T, 2//8)
    a3 = convert(T, 1//8)

    nxc, nyc, nzc = size(xc)
    m = A.interior

    @inbounds @batch for kc in 1:nzc-1
        k = (kc << 1) - 1
        for jc in 1:nyc-1
            j = (jc << 1) - 1
            for ic in 1:nxc-1
                i = (ic << 1) - 1

                x000 = xc[ic,jc,kc]
                x100 = xc[ic+1,jc,kc]
                x010 = xc[ic,jc+1,kc]
                x110 = xc[ic+1,jc+1,kc]
                x001 = xc[ic,jc,kc+1]
                x101 = xc[ic+1,jc,kc+1]
                x011 = xc[ic,jc+1,kc+1]
                x111 = xc[ic+1,jc+1,kc+1]

                x1 = x000 + x100
                x2 = x000 + x010

                x4 = x010 + x110
                x5 = x001 + x101
                x6 = x001 + x011

                if m[i,j,k]
                    x[i,j,k] += x000
                end

                if m[i+1,j,k]
                    x[i+1,j,k] = muladd(a1, x1, x[i+1,j,k])
                end

                if m[i,j+1,k]
                    x[i,j+1,k] = muladd(a1, x2, x[i,j+1,k])
                end

                if m[i+1,j+1,k]
                    xx = x1 + x4
                    x[i+1,j+1,k] = muladd(a2, xx, x[i+1,j+1,k])
                end

                if m[i,j,k+1]
                    xx = x000 + x001
                    x[i,j,k+1] = muladd(a1, xx, x[i,j,k+1])
                end

                if m[i+1,j,k+1]
                    xx = x1 + x5
                    x[i+1,j,k+1] = muladd(a2, xx, x[i+1,j,k+1])
                end

                if m[i,j+1,k+1]
                    xx = x2 + x6
                    x[i,j+1,k+1] = muladd(a2, xx, x[i,j+1,k+1])
                end

                if m[i+1,j+1,k+1]
                    xx = x1 + x4 + x5 + x011 + x111
                    x[i+1,j+1,k+1] = muladd(a3, xx, x[i+1,j+1,k+1])
                end
            end
        end
    end

    return x
end


restrict_size(n::Integer) = (n + 1) >> 1
restrict_size(ns::Tuple) = restrict_size.(ns)
restrict_size(ns...) = restrict_size(ns)

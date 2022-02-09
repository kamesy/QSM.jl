function expand(x::Tuple, n::Int)
    ntuple(i -> i > length(x) ? deepcopy(x[end]) : x[i], Val(n))
end

function expand(x::Tx, n::Int) where {Tx<:AbstractVector}
    Tx[i > length(x) ? deepcopy(x[end]) : x[i] for i in 1:n]
end

function expand(x, n::Int)
    [deepcopy(x) for _ in 1:n]
end

# Chebyshev nodes and interpolation properties.
# Reference: Trefethen, L.N. "Approximation Theory and Approximation Practice" SIAM, 2013, pp. 28-29.

# n Chebyshev nodes on [-1, 1]
# kind=1: roots of T_n(x);  kind=2: extrema of T_{n-1}(x)

using BPosits
using LinearAlgebra

function chebpts(n, kind=1; T=Float64)
    if kind == 1
        return T.([cos((2k - 1)*pi / (2n)) for k in 1:n])
    else
        return T.([cos((k - 1)*pi / (n - 1)) for k in 1:n])
    end
end

# Linear map from [-1, 1] to [c, d]
linscale(x, c, d) = @. c + (d - c) * (x + 1) / 2

# Geometric mean of all pairwise distances (log-sum to avoid overflow)
function meandistance(x)
    n = length(x)
    s = sum(log(abs(x[i] - x[j])) for i in 1:n for j in 1:n if i != j)
    exp(s / (n * (n - 1)))
end

n = 10
for T in (Float64, BPosit32, BPosit64)
    pts = chebpts(n; T=T)
    scaled = linscale(pts, T(0), T(pi))
    println("$T  meandist = ", meandistance(pts),
            "  monic bound = ", 2.0^(1 - n))
end

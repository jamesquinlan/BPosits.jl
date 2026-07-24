# Solve Sparse example
# ---------------------------------------------------------------------
# 1D Laplacian (tridiagonal, SPD) with known solution x = ones(n)
# ---------------------------------------------------------------------

using BPosits
using SparseArrays
using LinearAlgebra

n = 100
diag_vals = fill(2.0, n)
off_vals  = fill(-1.0, n - 1)
A64 = spdiagm(0 => diag_vals, 1 => off_vals, -1 => off_vals)
b64 = A64 * ones(n)

for T in (Float64, BPosit32, BPosit16)
    A = SparseMatrixCSC{T, Int64}(T.(A64))
    b = T.(b64)
    x = A \ b
    res = norm(Float64.(A) * Float64.(x) - Float64.(b))
    println("$T  ||Ax - b|| = $res")
end

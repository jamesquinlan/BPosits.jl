using BPosits
using LinearAlgebra

T = BPosit32

# Matrix from "A Collection of Test Matrices for Testing Computational Algorithms"
# by Gregory and Karney (p. 29). Inverse is known analytically.
A = T.([1.0 -2 3 1; -2 1 -2 -1; 3 -2 1 5; 1 -1 5 3])
LU = lu(A)

x = ones(T, 4)
b = T.([3, -4, 7, 8])

# Solve Ax = b
x_computed = A \ b
println("||Ax - b||       = ", norm(A * x_computed - b))
println("||x - x_exact||  = ", norm(x_computed - x))

# Solve using LU
x_computed_LU = LU \ b
println("||Ax_LU - b||    = ", norm(A * x_computed_LU - b))

# Known inverse (scaled by 1/52)
Ai = T(1/52) * T.([-15.0 -38 -1 -6; -38 -20 -6 16; -1 -6 -7 10; -6 16 10 8])
println("||A*Ai - I||     = ", norm(A * Ai - I))

# Solve via explicit inverse
x_computed_inv = Ai * b
println("||Ax_inv - b||   = ", norm(A * x_computed_inv - b))
println("||x_inv - x_exact|| = ", norm(x_computed_inv - x))

# Matrix properties (eigen/svd require Float64)
println("cond(A)          = ", cond(Float64.(A)))
println("norm(A, 2)       = ", norm(A, 2))

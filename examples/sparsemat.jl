using BPosits
using LinearAlgebra
using SparseArrays
using MatrixDepot

T = BPosit32

# ID = 6 is arc130 from the Harwell-Boeing collection
# 130 x 130 with condition number ~6e+10
# See: https://sparse.tamu.edu/HB/arc130
A = SparseMatrixCSC{T, Int64}(matrixdepot(sp(6)))
# or
# A = T.(matrixdepot(sp(6)))

# Random sparse matrix (without MatrixDepot)
B = sparse(rand(T, 10, 10))

# Sparse matrix from row/col/val triplets
C = sparse([1, 2, 3, 4], [1, 3, 4, 7], [T(1), T(2), T(3), T(4)], 10, 10)

# Random sparse with 40% nonzero density (generated in Float64, then cast)
D = T.(sprand(10, 10, 0.4)) + T(5) * I

# Convert a dense matrix to sparse
E = sparse(T.([4.0 1.0 0.0; 1.0 5.0 2.0; 0.0 2.0 6.0]))

# Using SparseMatrixCSC constructor directly
F = SparseMatrixCSC{T, Int64}(sparse(T.([4.0 1.0 0.0; 1.0 5.0 2.0; 0.0 2.0 6.0])))

# -------------------------------------------------------
# Condition number: sparse matrices must be converted to Float64
cond(Float64.(A), 1)          # 1-norm (or Inf-norm) for sparse
cond(Matrix(Float64.(A)))     # ~6e+10

# -------------------------------------------------------
# Special sparse matrix types
Diagonal(T.(diag(Matrix(A))))
LowerTriangular(T.(tril(Matrix(A))))
UpperTriangular(T.(triu(Matrix(A))))

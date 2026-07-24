# AMG Solve example
# ---------------------------------------------------------------------
# J.W. Ruge and K. Stuben, Algebraic multigrid, in Multigrid Methods,
#   vol. 3 of Frontiers in Applied Mathematics, SIAM, Philadelphia, PA,
#   1987, pp. 73-130.
# ---------------------------------------------------------------------

using BPosits
using AlgebraicMultigrid
using SparseArrays

T = BPosit32

A = poisson(T, 100)
b = A * ones(T, 100)
x = AlgebraicMultigrid.solve(A, b, RugeStubenAMG(); maxiter=3, abstol=1e-6)

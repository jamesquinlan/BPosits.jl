# AMG Preconditioner example
# ---------------------------------------------------------------------
# J.W. Ruge and K. Stuben, Algebraic multigrid, in Multigrid Methods,
#   vol. 3 of Frontiers in Applied Mathematics, SIAM, Philadelphia, PA,
#   1987, pp. 73-130.
# ---------------------------------------------------------------------

using BPosits
using AlgebraicMultigrid
import IterativeSolvers: cg

T = BPosit32

A  = poisson(T, 100)       # symmetric positive definite sparse matrix
ml = ruge_stuben(A)        # Ruge-Stuben solver
p  = aspreconditioner(ml)
x  = cg(A, A * ones(T, 100); Pl=p)

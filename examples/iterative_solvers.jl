using BPosits
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using IncompleteLU
using MatrixDepot

T = BPosit32
A = SparseMatrixCSC{T, Int64}(T.(matrixdepot(sp(6))))  # arc130 (Harwell-Boeing)
b = A * ones(T, size(A, 2))

x = jacobi(A, b; maxiter=100)
x = gauss_seidel(A, b; maxiter=100)
x = sor(A, b, T(1.25); maxiter=100)
x = gmres(A, b; restart=50, abstol=1e-6, maxiter=100)
x = cg(A, b; maxiter=100)

# ILU-preconditioned GMRES
P = ilu(A; τ=0.1)
x = gmres(A, b; Pl=P, maxiter=50, reltol=1e-8)
println("ILU-GMRES resid = ", norm(Float64.(A) * Float64.(x) - Float64.(b)))

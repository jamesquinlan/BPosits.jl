using BPosits, LinearAlgebra, Test

# Tests for LinearAlgebra compatibility with BPosit types.
# Most operations work natively; functions that need LAPACK (svd, eigvals, cond,
# nullspace) require a Float64 cast and are noted with comments.

@testset "LinearAlgebra LU" begin

    @testset "BPosit32 2x2" begin
        # [2 1; 4 3] x = [5; 11], exact solution x = [2, 1]
        A = BPosit32[2.0 1.0; 4.0 3.0]
        b = BPosit32[5.0, 11.0]
        F = lu(A)
        x = F \ b
        @test Float64.(x) ≈ [2.0, 1.0] atol=1e-6
        @test Float64.(F.L * F.U) ≈ Float64.(F.P * Matrix(A)) atol=1e-6
        @test Float64.(A * x .- b) ≈ zeros(2) atol=1e-6
    end

    @testset "BPosit32 3x3" begin
        A = BPosit32[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
        b = BPosit32[1.0, 2.0, 3.0]
        x = lu(A) \ b
        @test Float64.(A * x .- b) ≈ zeros(3) atol=1e-5
        F = lu(A)
        @test Float64.(F.L * F.U) ≈ Float64.(F.P * Matrix(A)) atol=1e-5
    end

    @testset "BPosit64 precision" begin
        A = BPosit64[2.0 1.0; 4.0 3.0]
        b = BPosit64[5.0, 11.0]
        @test Float64.(lu(A) \ b) ≈ [2.0, 1.0] atol=1e-14

        A3 = BPosit64[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
        b3 = BPosit64[1.0, 2.0, 3.0]
        @test Float64.(A3 * (lu(A3) \ b3) .- b3) ≈ zeros(3) atol=1e-13
    end

    @testset "BPosit16 basic" begin
        # [2 1; 1 3] x = [5; 7], solution x = [1.6, 1.8]
        A = BPosit16[2.0 1.0; 1.0 3.0]
        b = BPosit16[5.0, 7.0]
        x = lu(A) \ b
        @test Float64.(x) ≈ [1.6, 1.8] atol=1e-2   # ~3 decimal digits
    end

    @testset "Factorization reuse" begin
        A = BPosit32[3.0 1.0; 2.0 4.0]
        F = lu(A)
        b1 = BPosit32[4.0, 6.0]
        b2 = BPosit32[7.0, 10.0]
        @test Float64.(A * (F \ b1)) ≈ Float64.(b1) atol=1e-5
        @test Float64.(A * (F \ b2)) ≈ Float64.(b2) atol=1e-5
    end

    @testset "10x10 diagonally-dominant" begin
        rng_A = Float64[i == j ? 10.0 : 0.1 * sin(Float64(i * j)) for i in 1:10, j in 1:10]
        rng_b = [Float64(i) for i in 1:10]
        A = BPosit32.(rng_A)
        b = BPosit32.(rng_b)
        x = lu(A) \ b
        @test Float64.(x) ≈ rng_A \ rng_b rtol=1e-4
    end

end

@testset "LinearAlgebra functions (native)" begin
    # Symmetric positive definite test matrix
    A = BPosit32[3.0 1.0 0.5; 1.0 4.0 1.0; 0.5 1.0 5.0]
    v = BPosit32[1.0, 2.0, 3.0]

    @testset "Vector norms" begin
        @test Float64(norm(v, 1))   ≈ 6.0       atol=1e-6
        @test Float64(norm(v))      ≈ sqrt(14.0) atol=1e-5
        @test Float64(norm(v, Inf)) == 3.0
    end

    @testset "Matrix norms" begin
        # norm(A, p) treats matrix as a flat vector (Frobenius for p=2)
        @test Float64(norm(A))      ≈ sqrt(54.5) atol=1e-4  # Frobenius
        @test Float64(norm(A, Inf)) == 5.0                    # max element
        # Induced/operator norms
        @test Float64(opnorm(A, 1))   ≈ 6.5 atol=1e-5   # max col 1-norm
        @test Float64(opnorm(A, Inf)) ≈ 6.5 atol=1e-5   # max row 1-norm
        # opnorm(A, 2) and cond(A) need LAPACK -- cast to Float64:
        @test cond(Float64.(A)) ≈ 2.499 atol=1e-3
    end

    @testset "det and tr" begin
        @test Float64(det(A)) ≈ 52.0 atol=1e-3
        @test Float64(tr(A))  == 12.0
    end

    @testset "dot and cross" begin
        @test Float64(dot(v, v)) == 14.0
        w = BPosit32[1.0, 0.0, 0.0]
        @test Float64.(cross(v, w)) ≈ [0.0, 3.0, -2.0] atol=1e-6
    end

    @testset "Cholesky (SPD)" begin
        C = cholesky(A)
        @test Float64(det(C)) ≈ 52.0 atol=1e-3
        @test Float64.(C.L * C.U) ≈ Float64.(Matrix(A)) atol=1e-5
    end

    @testset "QR factorization" begin
        Q, R = qr(A)
        @test maximum(abs.(Float64.(Matrix(Q) * R .- A))) < 1e-5
        # Q is orthogonal
        @test Float64.(Matrix(Q)' * Matrix(Q)) ≈ I atol=1e-5
    end

    @testset "Solve (backslash)" begin
        b = BPosit32[1.0, 2.0, 3.0]
        x = A \ b
        @test maximum(abs.(Float64.(A * x .- b))) < 1e-6
    end

    @testset "Matrix predicates" begin
        @test issymmetric(A)
        @test isposdef(A)
        @test !issymmetric(BPosit32[1.0 2.0; 3.0 4.0])
    end

    @testset "Symmetric wrapper" begin
        B = Symmetric(A)
        @test Float64(opnorm(B, 1)) ≈ 6.5 atol=1e-5
        x = B \ BPosit32[1.0, 0.0, 0.0]
        @test Float64.(A * x) ≈ [1.0, 0.0, 0.0] atol=1e-5
    end

    @testset "eigvals / svd (via Float64 cast)" begin
        # These require LAPACK; cast to Float64 first
        ev = sort(eigvals(Float64.(A)))
        @test ev ≈ [2.377, 3.682, 5.940] atol=1e-2
        sv = svdvals(Float64.(A))
        @test maximum(sv) ≈ 5.940 atol=1e-2  # spectral norm = largest singular value
    end

end

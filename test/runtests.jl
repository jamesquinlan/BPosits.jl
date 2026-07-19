using BPosits
using LinearAlgebra
using Test

@testset "BPosits.jl Tests" begin

    @testset "Conversion and Range" begin
        @test Float64(BPosit32(1.0)) == 1.0
        @test Float64(BPosit32(2.0)) == 2.0
        @test Float64(BPosit32(0.5)) == 0.5
        @test Float64(BPosit32(0.0)) == 0.0
        
        # BPosit64 dynamic range target (~10^96)
        @test Float64(BPosit64(1e90)) > 0
        @test isfinite(BPosit64(1e90))
        @test isnan(BPosit64(NaN))
    end

    @testset "Basic Arithmetic" begin
        # BPosit32
        @test Float64(BPosit32(1.0) + BPosit32(2.0)) == 3.0
        @test Float64(BPosit32(5.0) - BPosit32(2.0)) == 3.0
        @test Float64(BPosit32(1.5) * BPosit32(2.0)) == 3.0
        @test Float64(BPosit32(6.0) / BPosit32(2.0)) == 3.0

        # BPosit64
        @test Float64(BPosit64(1.1) + BPosit64(2.2)) ≈ 3.3 atol=1e-10
    end

    @testset "Quire Dot Product" begin
        v1 = [BPosit32(1.0), BPosit32(2.0), BPosit32(3.0)]
        v2 = [BPosit32(1.0), BPosit32(0.5), BPosit32(1/3)]
        
        # Exact dot product [1, 2, 3] . [1, 0.5, 0.3333...]
        # 1*1 + 2*0.5 + 3*(1/3) = 1 + 1 + 1 = 3.0
        @test Float64(dot(v1, v2, quire=true)) == 3.0
        @test Float64(dot(v1, v2, quire=false)) ≈ 3.0
        
        # 64-bit Quire
        v1_64 = [BPosit64(1.0), BPosit64(2.0)]
        v2_64 = [BPosit64(1.5), BPosit64(0.5)]
        @test Float64(dot(v1_64, v2_64, quire=true)) == 2.5
    end

    @testset "BPosit8 Edge Cases" begin
        a = BPosit8(1.0)
        @test reinterpret(UInt8, a) == 0x40
        @test Float64(a + a) == 2.0
    end

    @testset "Comparisons and abs" begin
        @test BPosit32(1.0) < BPosit32(2.0)
        @test BPosit32(-3.0) < BPosit32(-2.0) < BPosit32(0.0) < BPosit32(2.5)
        @test BPosit32(2.0) <= BPosit32(2.0)
        @test BPosit32(2.0) == BPosit32(2.0)
        @test abs(BPosit32(-2.0)) == BPosit32(2.0)
        @test signbit(BPosit32(-1.5)) && !signbit(BPosit32(1.5))
        nar = BPosit32(NaN)
        @test !(nar < BPosit32(1.0)) && !(BPosit32(1.0) < nar) && !(nar == nar)
        @test isless(BPosit32(1.0), nar) && !isless(nar, BPosit32(1.0))  # NaR sorts last
        v = sort([BPosit16(3.0), BPosit16(-1.0), BPosit16(2.0)])
        @test Float64.(v) == [-1.0, 2.0, 3.0]
    end

    @testset "AbstractFloat Interface" begin
        # exhaustive: x == significand(x) * 2^exponent(x) for all BPosit16
        for i in 0x0001:0xffff
            x = reinterpret(BPosit16, i)
            isnan(x) && continue
            @test Float64(x) == Float64(significand(x)) * 2.0^exponent(x)
        end
        @test exponent(BPosit32(6.0)) == 2
        @test Float64(significand(BPosit32(6.0))) == 1.5
        fr, ex = frexp(BPosit32(6.0))
        @test Float64(fr) == 0.75 && ex == 3
        @test_throws DomainError exponent(BPosit32(0.0))

        @test precision(BPosit8) == 4
        @test precision(BPosit16) == 10
        @test precision(BPosit32) == 26
        @test precision(BPosit64) == 58
        @test precision(BPosit16(1.0)) == 10
        @test precision(floatmax(BPosit16)) == 4   # p_min + 1 at the cap

        @test Float64(round(BPosit32(2.5))) == 2.0   # ties to even
        @test Float64(floor(BPosit32(-1.5))) == -2.0
        @test Float64(ceil(BPosit32(1.2))) == 2.0
        @test Float64(trunc(BPosit32(-1.7))) == -1.0
        @test round(Int, BPosit32(2.6)) == 3
        @test Int(BPosit32(42.0)) == 42
        @test_throws InexactError Int(BPosit32(2.5))
        @test isinteger(BPosit16(7.0)) && !isinteger(BPosit16(7.5))

        # mixed-width promotion: wider wins, widening is exact
        @test BPosit16(1.5) + BPosit32(2.5) isa BPosit32
        @test Float64(BPosit16(1.5) + BPosit32(2.5)) == 4.0
        @test promote_type(BPosit8, BPosit64) == BPosit64
        @test widen(BPosit16) == BPosit32

        # BigFloat conversion is exact even beyond Float64 precision
        x64 = nextfloat(BPosit64(1.0))             # 1 + 2^-57
        @test BigFloat(x64) - 1 == big(2.0)^-57
        @test hash(BPosit32(1.5)) == hash(1.5)
        @test isfinite(BPosit32(1.0)) && !isinf(BPosit32(1.0)) && !isfinite(BPosit32(NaN))

        r = rand(BPosit32)
        @test r isa BPosit32 && 0.0 <= Float64(r) < 1.0
        @test length(rand(BPosit16, 3)) == 3
    end

    @testset "Fused Operations" begin
        # fma is single-rounded: the residual of a rounded product is nonzero
        a = BPosit16(0.1)
        r = a * a
        @test Float64(fma(a, a, -r)) == Float64(a)^2 - Float64(r)
        @test Float64(a * a + (-r)) == 0.0
        @test muladd(a, a, -r) == fma(a, a, -r)
        @test isnan(fma(BPosit32(NaN), BPosit32(1.0), BPosit32(1.0)))

        # fma agrees with its quire definition on random inputs
        for T in (BPosit16, BPosit32, BPosit64), _ in 1:1000
            x, y, z = T(2rand() - 1), T(2rand() - 1), T(2rand() - 1)
            q = Quire()
            BPosits.add_product!(q, x, y)
            BPosits.add_product!(q, z, one(T))
            @test fma(x, y, z) == T(q)
        end

        # exact summation survives catastrophic cancellation
        v = [BPosit64(1e30), BPosit64(1.0), BPosit64(-1e30)]
        @test Float64(sum(v)) == 1.0
        @test Float64(sum(BPosit32[])) == 0.0
        @test Float64(sum([BPosit8(1.0), BPosit8(2.0)])) == 3.0
        @test sum(view([BPosit32(1.0), BPosit32(2.0), BPosit32(3.0)], 1:2)) == BPosit32(3.0)
    end

    @testset "Width Conversions" begin
        # narrowing from BPosit64 must single-round: 1 + 2^-26 + 2^-57 lies
        # just above the tie between BPosit32(1.0) and its successor, but a
        # Float64 round trip first discards the 2^-57 and then breaks the
        # tie to even, landing on 1.0.
        x = nextfloat(BPosit64(1.0 + 2.0^-26))     # exact: 1 + 2^-26 + 2^-57
        @test BPosit32(x) == nextfloat(BPosit32(1.0))
        @test BPosit32(Float64(x)) == BPosit32(1.0)   # the double-rounding route

        # widening is exact; narrowing agrees with the Float64 route whenever
        # that route is itself exact (any source up to 32 bits)
        for i in 0x0001:0xffff
            p = reinterpret(BPosit16, i)
            isnan(p) && continue
            @test Float64(BPosit64(p)) == Float64(p)
            @test BPosit8(p) == BPosit8(Float64(p))
        end

        @test isnan(BPosit16(BPosit64(NaN)))
        @test Float64(BPosit8(BPosit32(0.0))) == 0.0
        @test BPosit32(BPosit64(1e80)) == floatmax(BPosit32)   # saturation
        @test BPosit32(BPosit64(-1e80)) == typemin(BPosit32)
    end

    @testset "Quire Matrix Multiply" begin
        # each element is exact-then-rounded: cancellation survives
        A = BPosit64[1e30 1.0 -1e30]          # 1x3
        x = [BPosit64(1.0), BPosit64(1.0), BPosit64(1.0)]
        @test Float64((A * x)[1]) == 1.0

        # agrees with the quire dot of the corresponding row/column slices
        for T in (BPosit16, BPosit32)
            A = T.(randn(7, 5)); B = T.(randn(5, 6))
            C = A * B
            @test C isa Matrix{T} && size(C) == (7, 6)
            for i in 1:7, j in 1:6
                @test C[i, j] == dot(A[i, :], B[:, j])
            end
            y = A * T.(randn(5))
            @test y isa Vector{T} && length(y) == 7
        end

        @test_throws DimensionMismatch mul!(zeros(BPosit32, 2, 2), BPosit32.(ones(2, 3)), BPosit32.(ones(2, 2)))
    end

    @testset "Elementary Functions" begin
        # the canonical smoke test
        @test isfinite(sin(BPosit8(pi)))
        @test Float64(sin(BPosit8(pi))) ≈ sin(Float64(BPosit8(pi))) atol=0.05

        for T in (BPosit16, BPosit32, BPosit64)
            tol = Dict(BPosit16 => 1e-2, BPosit32 => 1e-6, BPosit64 => 1e-12)[T]
            for x in (0.5, 1.0, 2.0, 3.0)
                @test Float64(sin(T(x))^2 + cos(T(x))^2) ≈ 1.0 atol=tol
                @test Float64(exp(log(T(x)))) ≈ x rtol=tol
                @test Float64(sqrt(T(x))^2) ≈ x rtol=tol
            end
        end
        @test Float64(exp2(BPosit64(0.5))) ≈ sqrt(2.0) rtol=1e-15
        @test Float64(hypot(BPosit32(3.0), BPosit32(4.0))) == 5.0
        @test Float64(BPosit32(2.0)^BPosit32(10.0)) == 1024.0
        @test Float64(atan(BPosit32(1.0), BPosit32(1.0))) ≈ pi/4 rtol=1e-6
        # sinpi/cospi: exact at (half-)integers, even for 8-bit
        for T in (BPosit8, BPosit16, BPosit32, BPosit64)
            @test Float64(sinpi(T(1.0))) == 0.0
            @test Float64(cospi(T(1.0))) == -1.0
            @test Float64(sinpi(T(0.5))) == 1.0
            @test Float64(cospi(T(2.0))) == 1.0
            @test Float64(sinpi(T(-0.5))) == -1.0
        end

        # domain errors return NaR, no exceptions (posit semantics)
        @test isnan(sqrt(BPosit32(-1.0)))
        @test isnan(log(BPosit32(-1.0)))
        @test isnan(sin(BPosit32(NaN)))
    end

    # include("gustafson_criteria.jl")
    # include("theorems.jl")
    include("lu_test.jl")
    include("aqua_test.jl")

end

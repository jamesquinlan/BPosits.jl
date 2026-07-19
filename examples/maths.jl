using BPosits

a = BPosit32(pi)
b = BPosit32(0.5)

println("sin(π)  = ", sin(a))
println("cos(π)  = ", cos(a))
println("tan(π)  = ", tan(a))
println("tan(0.5)= ", tan(b))

println("sinh(π) = ", sinh(a))
println("cosh(π) = ", cosh(a))
println("tanh(π) = ", tanh(a))

println("asin(0.5)= ", asin(b))
println("acos(0.5)= ", acos(b))
println("atan(0.5)= ", atan(b))

println("log2(π) = ", log2(a))
println("log10(π)= ", log10(a))
println("log1p(π)= ", log1p(a))

println("exp(π)  = ", exp(a))
println("exp2(π) = ", exp2(a))
println("exp10(π)= ", exp10(a))
println("expm1(π)= ", expm1(a))

println("sinpi(0.5) = ", sinpi(BPosit32(0.5)))   # exact: 1.0
println("cospi(1.0) = ", cospi(BPosit32(1.0)))   # exact: -1.0

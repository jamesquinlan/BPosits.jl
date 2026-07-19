"""
    BPosits

Bounded-posit arithmetic. Provides the `BPosit8`, `BPosit16`, `BPosit32`, and `BPosit64` floating-point types backed by the `libbposit` C library (`libbposit_jll`), together with an exact accumulator (`Quire`) and a fused `dot` product.

BPosit is a posit variant with a capped regime and full exponent field, giving every value at least `p_min` fraction bits and a uniform log-relative-error bound.
"""

module BPosits

using libbposit_jll: libbposit
using LinearAlgebra
import Random

import Base: AbstractFloat, Int, Int8, Int16, Int32, Int64, Integer,
             reinterpret, precision, frexp, exponent, significand,
             +, -, *, /, ^, ==, <, <=, isless, abs, signbit, isnan, isinf, isfinite,
             show, Float32, Float64, typemin, typemax, floatmin, floatmax, eps,
             sqrt, cbrt, exp, exp2, exp10, expm1, log, log2, log10, log1p,
             sin, cos, tan, asin, acos, atan, sinpi, cospi,
             sinh, cosh, tanh, asinh, acosh, atanh, hypot

import LinearAlgebra: dot

export BPosit8, BPosit16, BPosit32, BPosit64, AnyBPosit, Quire, dot, printbits

"""
    AnyBPosit <: AbstractFloat

Abstract supertype of the four bounded-posit formats. The standard parameters
are `es = min(4, n ÷ 4)` and `k_max = 1, 7, 13, 19` for `n = 8, 16, 32, 64`,
giving fraction floors `p_min = 3, 3, 13, 39`.
"""
abstract type AnyBPosit <: AbstractFloat end

primitive type BPosit8  <: AnyBPosit 8 end
primitive type BPosit16 <: AnyBPosit 16 end
primitive type BPosit32 <: AnyBPosit 32 end
primitive type BPosit64 <: AnyBPosit 64 end

const posit_types = [
    (:BPosit8,  :Int8,   :UInt8,  8),
    (:BPosit16, :Int16,  :UInt16, 16),
    (:BPosit32, :Int32,  :UInt32, 32),
    (:BPosit64, :Int64,  :UInt64, 64),
]

for (T, ST, UT, N) in posit_types
    es = N == 8 ? 2 : 4
    kmax = Dict(8 => 1, 16 => 7, 32 => 13, 64 => 19)[N]
    @eval Base.reinterpret(::Type{$UT}, x::$T) = Base.bitcast($UT, x)
    @eval Base.reinterpret(::Type{$T}, x::$UT) = Base.bitcast($T, x)
    @eval Base.reinterpret(::Type{$ST}, x::$T) = Base.bitcast($ST, x)
    @eval Base.reinterpret(::Type{$T}, x::$ST) = Base.bitcast($T, x)
    @eval Base.uinttype(::Type{$T}) = $UT
    @eval Base.inttype(::Type{$T}) = $ST
    @eval _es(::Type{$T}) = $es
    @eval _kmax(::Type{$T}) = $kmax
end

# Decode sign, total binary exponent E, and the 61-bit significand (hidden
# bit at position 60) directly from the bit pattern. Float64 round-trips are
# not faithful for BPosit64 (58 fovea bits > 53), so introspection must be
# bit-exact. Caller excludes zero and NaR.
function _bitfields(x::T) where T <: AnyBPosit
    n = 8 * sizeof(T)
    es = _es(T)
    kmax = _kmax(T)
    u = reinterpret(Base.uinttype(T), x)
    neg = signbit(x)
    neg && (u = -u)
    w = UInt64(u) << (64 - n + 1)
    rbit = w >>> 63 == 1
    cap = rbit ? kmax + 1 : kmax
    run = min(rbit ? leading_ones(w) : leading_zeros(w), cap)
    k = rbit ? run - 1 : -run
    consumed = run + (run < cap ? 1 : 0)
    rest = w << consumed
    e = Int(rest >>> (64 - es))
    frac = (UInt64(1) << 60) | ((rest << es) >>> 4)
    return neg, k * (1 << es) + e, frac
end

"""
    Quire()

A 2048-bit fixed-point exact accumulator (32 words of 64 bits, the `2^0`
position at word 16). One quire serves all BPosit widths: products of two
BPosit64 values fit with at least 383 carry bits. Accumulate with
[`add_product!`](@ref) / [`sub_product!`](@ref), reset with [`clear!`](@ref),
and round back with `BPositN(q)`.
"""
mutable struct Quire
    words::NTuple{32, UInt64}
    function Quire()
        q = new(ntuple(_ -> 0x0, 32))
        ccall((:bposit_quire_clear, libbposit), Cvoid, (Ref{Quire},), q)
        return q
    end
end

"""
    clear!(q::Quire)

Reset the accumulator to zero.
"""
clear!(q::Quire) = ccall((:bposit_quire_clear, libbposit), Cvoid, (Ref{Quire},), q)

"""
    add_product!(q::Quire, a, b)

Accumulate `+a*b` into the quire exactly (no rounding).
"""
function add_product! end

"""
    sub_product!(q::Quire, a, b)

Accumulate `-a*b` into the quire exactly (no rounding).
"""
function sub_product! end

for (T, N) in [(:BPosit8, 8), (:BPosit16, 16), (:BPosit32, 32), (:BPosit64, 64)]
    add_c = Symbol("bposit_quire_add_product_f", N)
    sub_c = Symbol("bposit_quire_sub_product_f", N)
    to_p  = Symbol("bposit_quire_to_bposit", N)
    @eval add_product!(q::Quire, a::$T, b::$T) =
        ccall(($(QuoteNode(add_c)), libbposit), Cvoid, (Ref{Quire}, $T, $T), q, a, b)
    @eval sub_product!(q::Quire, a::$T, b::$T) =
        ccall(($(QuoteNode(sub_c)), libbposit), Cvoid, (Ref{Quire}, $T, $T), q, a, b)
    @eval (::Type{$T})(q::Quire) =
        ccall(($(QuoteNode(to_p)), libbposit), $T, (Ref{Quire},), q)
end

for T in [:BPosit8, :BPosit16, :BPosit32, :BPosit64]
    for (op, c_name) in [(:+, "add"), (:-, "sub"), (:*, "mul"), (:/, "div")]
        func_name = Symbol(lowercase(string(T)), "_", c_name)
        @eval Base.$op(a::$T, b::$T) =
            ccall(($(QuoteNode(func_name)), libbposit), $T, ($T, $T), a, b)
    end
    neg_c = Symbol(lowercase(string(T)), "_neg")
    @eval Base.:-(a::$T) = ccall(($(QuoteNode(neg_c)), libbposit), $T, ($T,), a)
end

for T in [:BPosit8, :BPosit16, :BPosit32, :BPosit64]
    from_c = Symbol(lowercase(string(T)), "_from_double")
    to_c   = Symbol(lowercase(string(T)), "_to_double")
    @eval (::Type{$T})(f::Float64) =
        ccall(($(QuoteNode(from_c)), libbposit), $T, (Float64,), f)
    @eval (::Type{$T})(f::Float32) = $T(Float64(f))
    @eval Float64(p::$T) = ccall(($(QuoteNode(to_c)), libbposit), Float64, ($T,), p)
    @eval Float32(p::$T) = Float32(Float64(p))
end

(::Type{T})(f::Real) where T <: AnyBPosit = T(Float64(f))
(::Type{T})(f::BigFloat) where T <: AnyBPosit = T(Float64(f))

# Width conversions go through the C library: a single rounding, where the
# Float64 route would double round for BPosit64 sources.
for (Ta, _, _, Na) in posit_types, (Tb, _, _, Nb) in posit_types
    Na == Nb && continue
    conv_c = Symbol("bposit", Na, "_to_bposit", Nb)
    @eval (::Type{$Tb})(x::$Ta) =
        ccall(($(QuoteNode(conv_c)), libbposit), $Tb, ($Ta,), x)
end

for (T, N) in [(:BPosit8, 8), (:BPosit16, 16), (:BPosit32, 32), (:BPosit64, 64)]
    dot_c = Symbol("bposit", N, "_dot")
    @eval _dot_quire(a::DenseVector{$T}, b::DenseVector{$T}) =
        ccall(($(QuoteNode(dot_c)), libbposit), $T, (Ptr{$T}, Ptr{$T}, Csize_t),
              a, b, length(a))
end

# muladd aliases fma.
for (T, N) in [(:BPosit8, 8), (:BPosit16, 16), (:BPosit32, 32), (:BPosit64, 64)]
    fma_c = Symbol("bposit", N, "_fma")
    sum_c = Symbol("bposit", N, "_sum")
    @eval Base.fma(a::$T, b::$T, c::$T) =
        ccall(($(QuoteNode(fma_c)), libbposit), $T, ($T, $T, $T), a, b, c)
    @eval Base.muladd(a::$T, b::$T, c::$T) = fma(a, b, c)
    @eval _sum_quire(a::DenseVector{$T}) =
        ccall(($(QuoteNode(sum_c)), libbposit), $T, (Ptr{$T}, Csize_t), a, length(a))
end

"""
    sum(a::AbstractVector{T}) where T <: AnyBPosit

Sum of a BPosit vector, accumulated exactly in a quire and rounded once.
"""
Base.sum(a::DenseVector{T}) where T <: AnyBPosit = _sum_quire(a)
Base.sum(a::AbstractVector{T}) where T <: AnyBPosit =
    (q = Quire(); o = one(T); for x in a; add_product!(q, x, o); end; T(q))

# A*B and A*x dispatch here.
for (T, N) in [(:BPosit8, 8), (:BPosit16, 16), (:BPosit32, 32), (:BPosit64, 64)]
    mm_c = Symbol("bposit", N, "_matmul")
    @eval function LinearAlgebra.mul!(C::Matrix{$T}, A::Matrix{$T}, B::Matrix{$T})
        m, k = size(A)
        size(B, 1) == k || throw(DimensionMismatch("A has $k columns, B has $(size(B, 1)) rows"))
        size(C) == (m, size(B, 2)) || throw(DimensionMismatch("C has size $(size(C)), expected $((m, size(B, 2)))"))
        ccall(($(QuoteNode(mm_c)), libbposit), Cvoid,
              (Ptr{$T}, Ptr{$T}, Ptr{$T}, Csize_t, Csize_t, Csize_t),
              C, A, B, m, k, size(B, 2))
        return C
    end
    @eval function LinearAlgebra.mul!(y::Vector{$T}, A::Matrix{$T}, x::Vector{$T})
        m, k = size(A)
        length(x) == k || throw(DimensionMismatch("A has $k columns, x has length $(length(x))"))
        length(y) == m || throw(DimensionMismatch("y has length $(length(y)), expected $m"))
        ccall(($(QuoteNode(mm_c)), libbposit), Cvoid,
              (Ptr{$T}, Ptr{$T}, Ptr{$T}, Csize_t, Csize_t, Csize_t),
              y, A, x, m, k, 1)
        return y
    end
end

"""
    dot(a::AbstractVector{T}, b::AbstractVector{T}; quire=true) where T <: AnyBPosit

Dot product of BPosit vectors. With `quire = true` (the default) the products
are accumulated exactly in a quire and rounded once; for dense vectors this is
a single allocation-free call into the C kernel. With `quire = false` each
product and partial sum is rounded, for comparison studies.
"""
function LinearAlgebra.dot(a::AbstractVector{T}, b::AbstractVector{T}; quire::Bool=true) where T <: AnyBPosit
    length(a) == length(b) ||
        throw(DimensionMismatch("dot: lengths $(length(a)) and $(length(b)) differ"))
    if quire
        if a isa DenseVector{T} && b isa DenseVector{T}
            return _dot_quire(a, b)
        end
        q = Quire()
        for i in eachindex(a, b)
            add_product!(q, a[i], b[i])
        end
        return T(q)
    else
        acc = T(0.0)
        for i in eachindex(a, b)
            acc = acc + a[i] * b[i]
        end
        return acc
    end
end

# Bit patterns ordered as two's-complement integers are value-ordered
# (Gustafson criterion 6), so comparisons reduce to integer comparisons.
# NaR compares like NaN: `==`/`<` are false, `isless` sorts it last.
for T in [:BPosit8, :BPosit16, :BPosit32, :BPosit64]
    @eval begin
        Base.signbit(x::$T) = reinterpret(Base.inttype($T), x) < 0
        Base.:(==)(a::$T, b::$T) = !isnan(a) && !isnan(b) &&
            reinterpret(Base.uinttype($T), a) == reinterpret(Base.uinttype($T), b)
        Base.:<(a::$T, b::$T) = !isnan(a) && !isnan(b) &&
            reinterpret(Base.inttype($T), a) < reinterpret(Base.inttype($T), b)
        Base.:<=(a::$T, b::$T) = !isnan(a) && !isnan(b) &&
            reinterpret(Base.inttype($T), a) <= reinterpret(Base.inttype($T), b)
        Base.isless(a::$T, b::$T) = isnan(a) ? false : (isnan(b) ? true :
            reinterpret(Base.inttype($T), a) < reinterpret(Base.inttype($T), b))
        Base.abs(x::$T) = signbit(x) ? -x : x
    end
end

function Base.exponent(x::T) where T <: AnyBPosit
    (isnan(x) || iszero(x)) &&
        throw(DomainError(x, "`exponent` requires a finite nonzero value"))
    return _bitfields(x)[2]
end

function Base.significand(x::T) where T <: AnyBPosit
    (isnan(x) || iszero(x)) && return x
    neg, _, frac = _bitfields(x)
    n = 8 * sizeof(T)
    es = _es(T)
    p0 = n - 3 - es
    # Rebuild at regime k = 0, exponent 0: the fovea fraction budget p0
    # dominates every other regime, so this is always exact.
    explicit = (frac & ((UInt64(1) << 60) - 1)) >> (60 - p0)
    u = (UInt64(0b10) << (es + p0)) | explicit
    r = reinterpret(T, Base.uinttype(T)(u))
    return neg ? -r : r
end

function Base.frexp(x::T) where T <: AnyBPosit
    (isnan(x) || iszero(x)) && return x, 0
    return significand(x) / T(2), exponent(x) + 1
end

Base.precision(::Type{T}) where T <: AnyBPosit = 8 * sizeof(T) - 2 - _es(T)
Base.precision(x::T) where T <: AnyBPosit = Int(get_decode_info(x).fraction_len) + 1

# Rounding routes through Float64, which is exact for widths up to 32 bits.
# (BPosit64 values in (2^53, 2^58) have sub-integer spacing and may double
# round here; a bit-exact path can replace this if that regime matters.)
Base.round(x::T, r::RoundingMode) where T <: AnyBPosit = T(round(Float64(x), r))
Base.round(::Type{I}, x::AnyBPosit, r::RoundingMode=RoundNearest) where I <: Integer =
    round(I, Float64(x), r)
Base.unsafe_trunc(::Type{I}, x::AnyBPosit) where I <: Integer = unsafe_trunc(I, Float64(x))

function (::Type{I})(x::AnyBPosit) where I <: Integer
    isinteger(x) || throw(InexactError(Symbol(I), I, x))
    return I(Float64(x))
end

Base.isinteger(x::AnyBPosit) = isfinite(x) && x == round(x, RoundToZero)
Base.isinf(::AnyBPosit) = false
Base.isfinite(x::AnyBPosit) = !isnan(x)

# (num, pow, den) with x == num * 2^pow / den, for Base's Real hashing.
function Base.decompose(x::AnyBPosit)
    isnan(x) && return 0, 0, 0
    iszero(x) && return 0, 0, 1
    neg, E, frac = _bitfields(x)
    return neg ? -Int64(frac) : Int64(frac), E - 60, 1
end

function Base.BigFloat(x::AnyBPosit)
    isnan(x) && return BigFloat(NaN)
    iszero(x) && return BigFloat(0)
    neg, E, frac = _bitfields(x)
    r = ldexp(BigFloat(frac), E - 60)
    return neg ? -r : r
end

Base.widen(::Type{BPosit8})  = BPosit16
Base.widen(::Type{BPosit16}) = BPosit32
Base.widen(::Type{BPosit32}) = BPosit64
Base.widen(::Type{BPosit64}) = BigFloat

# Wider format wins; widening is exact.
for i in eachindex(posit_types), j in eachindex(posit_types)
    i == j && continue
    Ta, Tb = posit_types[i][1], posit_types[j][1]
    W = posit_types[max(i, j)][1]
    @eval Base.promote_rule(::Type{$Ta}, ::Type{$Tb}) = $W
end

Random.rand(rng::Random.AbstractRNG, ::Random.SamplerTrivial{Random.CloseOpen01{T}}) where T <: AnyBPosit =
    T(rand(rng, Float64))

# Elementary functions evaluate in the C library: double for 8/16/32-bit,
# long double for 64-bit (whose 58-bit fovea exceeds double precision).
# Domain errors return NaR rather than throwing.
const UNARY_MATH = [:sqrt, :cbrt, :exp, :exp2, :exp10, :expm1,
                    :log, :log2, :log10, :log1p,
                    :sin, :cos, :tan, :asin, :acos, :atan, :sinpi, :cospi,
                    :sinh, :cosh, :tanh, :asinh, :acosh, :atanh]

for T in [:BPosit8, :BPosit16, :BPosit32, :BPosit64]
    for fn in UNARY_MATH
        c_name = Symbol(lowercase(string(T)), "_", fn)
        @eval Base.$fn(x::$T) = ccall(($(QuoteNode(c_name)), libbposit), $T, ($T,), x)
    end
    for (fn, c_suffix) in [(:^, "pow"), (:atan, "atan2"), (:hypot, "hypot")]
        c_name = Symbol(lowercase(string(T)), "_", c_suffix)
        @eval Base.$fn(a::$T, b::$T) =
            ccall(($(QuoteNode(c_name)), libbposit), $T, ($T, $T), a, b)
    end
end

function Base.typemin(::Type{T}) where T <: AnyBPosit
    u = Base.uinttype(T)
    nbits = 8 * sizeof(T)
    reinterpret(T, (u(1) << (nbits - 1)) + u(1))  # NaR + 1: most negative finite
end

function Base.typemax(::Type{T}) where T <: AnyBPosit
    u = Base.uinttype(T)
    nbits = 8 * sizeof(T)
    reinterpret(T, (u(1) << (nbits - 1)) - u(1))  # NaR - 1: maxpos
end

Base.floatmin(::Type{T}) where T <: AnyBPosit = nextfloat(T(0.0))
Base.floatmax(::Type{T}) where T <: AnyBPosit = typemax(T)

# Spacing varies by regime; report eps at one, where the fovea is widest.
Base.eps(::Type{T}) where T <: AnyBPosit = nextfloat(one(T)) - one(T)

function Base.isnan(x::T) where T <: AnyBPosit
    u = Base.uinttype(T)
    nbits = 8 * sizeof(T)
    reinterpret(u, x) == u(1) << (nbits - 1)  # NaR is 100...0
end

Base.show(io::IO, x::AnyBPosit) = print(io, isnan(x) ? "NaR" : Float64(x))
Base.promote_rule(::Type{T}, ::Type{S}) where {T <: AnyBPosit, S <: Real} = T

function Base.nextfloat(x::T) where T <: AnyBPosit
    isnan(x) && return x
    u = Base.uinttype(T)
    reinterpret(T, reinterpret(u, x) + u(1))
end

function Base.prevfloat(x::T) where T <: AnyBPosit
    isnan(x) && return x
    u = Base.uinttype(T)
    reinterpret(T, reinterpret(u, x) - u(1))
end

Base.bitstring(x::AnyBPosit) = bitstring(reinterpret(Base.uinttype(typeof(x)), x))

struct DecodeInfo
    nbits::Int32
    sign_len::Int32
    regime_len::Int32
    exponent_len::Int32
    fraction_len::Int32
end

for (T, N) in [(:BPosit8, 8), (:BPosit16, 16), (:BPosit32, 32), (:BPosit64, 64)]
    func_name = Symbol("bposit_decode_info_", N)
    @eval function get_decode_info(x::$T)
        info = Ref{DecodeInfo}()
        ccall(($(QuoteNode(func_name)), libbposit), Cvoid, ($T, Ref{DecodeInfo}), x, info)
        return info[]
    end
end

"""
    printbits(x::AnyBPosit)

Print the bit string of `x` with fields colored: sign (red), regime (yellow),
exponent (pink), fraction (green).
"""
function printbits(x::T) where T <: AnyBPosit
    s = bitstring(x)
    info = get_decode_info(x)
    red, yellow, pink, green, reset = "\033[31m", "\033[33m", "\033[95m", "\033[32m", "\033[0m"
    p = 1
    print(red, s[p:p+info.sign_len-1])
    p += info.sign_len
    print(yellow, s[p:p+info.regime_len-1])
    p += info.regime_len
    if info.exponent_len > 0
        print(pink, s[p:p+info.exponent_len-1])
        p += info.exponent_len
    end
    if info.fraction_len > 0
        print(green, s[p:p+info.fraction_len-1])
    end
    println(reset)
end

end # module

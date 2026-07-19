# BPosits.jl

[![CI](https://github.com/jamesquinlan/BPosits.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jamesquinlan/BPosits.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/jamesquinlan/BPosits.jl/graph/badge.svg?token=KO0KVNKYSP)](https://codecov.io/github/jamesquinlan/BPosits.jl)
<!-- 
[![Aqua](https://img.shields.io/badge/Aqua-tested-brightgreen)](...)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT) -->


![Scientific Computing](https://img.shields.io/badge/Scientific%20Computing-Julia-blue)
![Numerical Computing](https://img.shields.io/badge/Numerical-Computing-success)
​![Mixed Precision](https://img.shields.io/badge/Mixed-Precision-red)

BPosits.jl is a Julia package implementing **Bounded Posits (BPosit)** -- a posit variant with a capped regime and a guaranteed precision floor.

## Key Features (standard v2)

- **Bounded Regime**: |k| ≤ k_max = 1 / 7 / 13 / 19 for 8/16/32/64 bits, eliminating the precision collapse of standard posits.
- **Exponent Size**: es(n) = min(4, n/4) — 2 bits for BPosit8, 4 bits for all wider formats; the full exponent field is present at every regime.
- **Precision Floor**: p_min = n − 2 − es − k_max = 3 / 3 / 13 / 39 fraction bits guaranteed everywhere, giving a uniform log-relative-error bound λ ≤ ln(1 + 2^−(p_min+1)).
- **Exact Quire**: a 2048-bit fixed-point accumulator provides exact dot products for all widths; `dot(a, b; quire=true)`.
- **Lossless casting**: BPosit8 → Float16, BPosit16 → Float32, BPosit32 → Float64 are exact; BPosit16 spans Float32's full dynamic range at half the width.

This package is modeled after [Posits.jl](https://github.com/takum-arithmetic/Posits.jl) and uses the [`libbposit`](https://github.com/jamesquinlan/libbposit) C library (via `libbposit_jll`) for all numerics.

## The Format

BPosits implement the BPosit format family (8/16/32/64-bit) with per-width parameters es = min(4, n/4) and k_max = 1/7/13/19, where the regime cap and fraction floor are **dual** constraints:

```
p_min = n − 2 − es − k_max = 3 / 3 / 13 / 39
```

The full es-bit exponent field is present at every regime (no truncation), which makes every bit pattern canonical and yields the uniform log-relative-error bound λ ≤ ln(1 + 2^−(p_min+1)) across the entire dynamic range. Values beyond the regime cap saturate to maxpos/minpos, matching Posit Standard (2022) rounding semantics (round-to-nearest, ties to even bit pattern; never to 0 or NaR).

## Implementation

The C backend ([`libbposit`](https://github.com/jamesquinlan/libbposit), ISO C99) provides:

- **CLZ/shift codec**: regime, exponent, and fraction fields are each decoded/encoded in O(1), no per-bit loops.
- **Single-rounded direct arithmetic** — add/sub/mul align the 61-bit significands in a 128-bit integer with sticky-bit (and borrow-aware) tracking, so the final encode performs the *only* rounding. ~19 ns/op, roughly 8× faster than the reference posit library at -O3, and verified bit-identical to the exact quire reference over exhaustive 8-bit pairs and 10⁷ random pairs per width.
- **A single 2048-bit quire shared by all widths** (BPosit64 products retain ≥383 carry bits) with fused `dot`, `fma`, `sum`, and `mul!` (matrix multiply) kernels that accumulate exactly on the C stack and round once — zero allocations from Julia.
- **Single-rounded conversions**: from Float64 via the IEEE bit pattern, and between BPosit widths by decode/re-encode (widening is exact; narrowing avoids the double rounding a Float64 round trip would introduce for BPosit64 sources).
- **Elementary functions** (`sin`, `exp`, `log`, ...) via double (8/16/32-bit) or x87 long double (64-bit, whose 58-bit significand exceeds double), plus exact `sinpi`/`cospi`. Domain errors return NaR; no exceptions.

The Julia layer defines the four widths as primitive subtypes of `AbstractFloat` with the complete numeric interface: comparisons (two's-complement bit-pattern order), rounding and integer conversion, `exponent`/`significand`/`frexp`/`precision`, hashing, exact `BigFloat` conversion, wider-wins promotion between widths, and `rand`.

## Installation and Usage

```julia
] add BPosits
```

```julia
using BPosits

x = BPosit32(1.0)
y = BPosit32(2.0)
z = x + y
printbits(z)                    # color-coded sign | regime | exponent | fraction
dot([x, y], [y, x], quire=true) # exact accumulation
```

## License

MIT License. See LICENSE for details.

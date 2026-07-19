# BPosits.jl

**Bounded Posit arithmetic for Julia.**

BPosits implements the BPosit format family — a posit variant with a capped regime and a guaranteed precision floor — backed by the [`libbposit`](https://github.com/jamesquinlan/libbposit) C library.

## Installation

```julia
] add BPosits
```

## Format summary

| Type       | `n` | `es` | `k_max` | `p_min` | Dynamic range (decades) |
|------------|-----|------|---------|---------|------------------------|
| `BPosit8`  |  8  |  2   |   1     |   3     | ±2.4                   |
| `BPosit16` | 16  |  4   |   7     |   3     | ±38.5                  |
| `BPosit32` | 32  |  4   |  13     |  13     | ±67.4                  |
| `BPosit64` | 64  |  4   |  19     |  39     | ±96.3                  |

`p_min = n − 2 − es − k_max` fraction bits are guaranteed at every value, giving a uniform log-relative-error bound `λ ≤ ln(1 + 2^−(p_min+1))` across the full dynamic range.

## Quick start

```julia
using BPosits

x = BPosit32(1.5)
y = BPosit32(2.25)

x + y          # 3.75
x * y          # 3.375
sqrt(x)        # 1.2247...
printbits(x)   # color-coded sign | regime | exponent | fraction

# Exact dot product via 2048-bit quire
a = BPosit32.([1.0, 2.0, 3.0])
b = BPosit32.([4.0, 5.0, 6.0])
dot(a, b)              # 32.0, exact
dot(a, b; quire=false) # rounded accumulation, for comparison
```

## Key features

- **Bounded regime**: |k| ≤ k_max eliminates precision collapse at extreme values.
- **Full exponent field**: all `es` bits are always present — no truncation, every bit pattern canonical.
- **Exact quire**: a 2048-bit fixed-point accumulator for `dot`, `fma`, `sum`, and matrix multiply with a single final rounding.
- **Lossless widening**: `BPosit8 → Float16`, `BPosit16 → Float32`, `BPosit32 → Float64` are exact.
- **Elementary functions**: `sin`, `exp`, `log`, `sqrt`, … via double (8/16/32-bit) or x87 long double (64-bit). Domain errors return NaR.

## See also

- [libbposit](https://github.com/jamesquinlan/libbposit) — the C backend
- [API Reference](@ref)

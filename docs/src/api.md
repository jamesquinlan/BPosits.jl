# API Reference

```@docs
BPosits
```

## Types

```@docs
AnyBPosit
```

The four concrete types — `BPosit8`, `BPosit16`, `BPosit32`, `BPosit64` — are primitive
subtypes of `AnyBPosit` with widths 8, 16, 32, and 64 bits respectively. They support
the full Julia numeric interface: arithmetic, comparisons, conversions, rounding,
`exponent`, `significand`, `frexp`, `precision`, `eps`, `rand`, and `hash`.

## Quire (exact accumulator)

```@docs
Quire
BPosits.clear!
BPosits.add_product!
BPosits.sub_product!
```

## Fused operations

```@docs
dot
BPosits.sum
```

## Utilities

```@docs
printbits
```

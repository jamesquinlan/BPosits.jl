# Contributing to BPosits.jl

Thank you for your interest. BPosits.jl implements the BPosit bounded-posit arithmetic standard. The C backend is maintained as a separate library at [github.com/jamesquinlan/libbposit](https://github.com/jamesquinlan/libbposit) and delivered automatically via `libbposit_jll` — no C compiler or build step is needed to work on this package.

## Prerequisites

- Julia 1.10 or later

## Running the tests

```sh
julia --project=. test/runtests.jl
```

The suite runs ~131 k assertions in under two seconds. All tests must pass before a pull request will be merged.

## What to contribute

- **Bug reports** — open a GitHub issue with the Julia version, OS, and a minimal reproducer.
- **Correctness fixes** — for the Julia wrapper (`src/BPosits.jl`), include a test that would have caught the bug. For C-level fixes, open an issue or PR in the [libbposit repo](https://github.com/jamesquinlan/libbposit).
- **New tests** — additional coverage for edge cases, wider types, or operations not yet exercised.
- **Performance improvements** — must not regress correctness and should include a benchmark showing the gain.
- **Documentation** — clarifications to `README.md` or docstrings in `src/BPosits.jl`.

## What not to change without discussion

- The standard parameters (`es`, `k_max`, `p_min`) — changing them invalidates the exhaustive theorems.
- The quire layout (2048-bit, word 16 at 2⁰) — co-designed with the 64-bit regime cap.
- Rounding semantics — must remain round-to-nearest, ties to even, saturate to maxpos/minpos.

Open an issue first if you want to discuss any of the above.

## Code style

Follow the [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/). No banner comments; docstrings on exported symbols only. Match the surrounding code style rather than reformatting unrelated lines.

## Pull request checklist

- [ ] `julia --project=. test/runtests.jl` passes (all tests green)
- [ ] New behavior is covered by at least one test
- [ ] `Project.toml` version is bumped if the public API changes

## License

By contributing you agree that your changes will be released under the [MIT License](LICENSE).

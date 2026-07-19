using Test

# Aqua is a test-only dep ([extras]); available via `Pkg.test()` but not when
# running runtests.jl directly. Skip gracefully in the latter case.
@testset "Aqua" begin
    if Base.find_package("Aqua") !== nothing
        import Aqua
        Aqua.test_all(
            Base.require(Base.PkgId(Base.UUID("8a7cb713-368b-48d9-84fe-cff0fadda989"), "BPosits"));
            ambiguities      = false,   # false-positives from Base promotions on AbstractFloat subtypes
            stale_deps       = true,
            deps_compat      = true,
            piracy           = true,
            undefined_exports = true,
        )
    else
        @info "Aqua not in current environment; run `Pkg.test()` for quality-assurance checks"
    end
end

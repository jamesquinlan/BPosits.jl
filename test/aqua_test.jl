using Test

@testset "Aqua" begin
    if Base.find_package("Aqua") !== nothing
        import Aqua
        Aqua.test_all(BPosits; ambiguities = false)
    else
        @info "Aqua not in current environment; run `Pkg.test()` for quality-assurance checks"
    end
end

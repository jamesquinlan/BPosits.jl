using Documenter
using BPosits

makedocs(
    sitename = "BPosits.jl",
    authors  = "James Quinlan",
    modules  = [BPosits],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://jamesquinlan.github.io/BPosits.jl/stable/",
    ),
    pages = [
        "Home"      => "index.md",
        "API"       => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo       = "github.com/jamesquinlan/BPosits.jl.git",
    devbranch  = "main",
)

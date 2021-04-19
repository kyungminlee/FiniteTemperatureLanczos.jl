using Documenter
using QuantumHamiltonian

makedocs(
    modules=[FiniteTemperatureLanczos],
    doctest=true,
    sitename="FiniteTemperatureLanczos.jl",
    format=Documenter.HTML(prettyurls=!("local" in ARGS)),
    authors="Kyungmin Lee",
    checkdocs=:all,
    pages = [
        "Home" => "index.md",
    ]
  )

deploydocs(;
    repo="github.com/kyungminlee/FiniteTemperatureLanczos.jl.git",
    devbranch="dev",
)

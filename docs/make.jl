using QSM
using Documenter

DocMeta.setdocmeta!(QSM, :DocTestSetup, :(using QSM); recursive=true)

makedocs(;
    modules=[QSM],
    authors="kamesy <ckames@physics.ubc.ca>",
    repo="https://github.com/kamesy/QSM.jl/blob/{commit}{path}#{line}",
    sitename="QSM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kamesy.github.io/QSM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/kamesy/QSM.jl",
    devbranch="main",
)

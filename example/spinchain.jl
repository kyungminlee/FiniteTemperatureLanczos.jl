using LatticeTools
using QuantumHamiltonian
using ProgressMeter
using LinearAlgebra
using KrylovKit
using Plots

using FiniteTemperatureLanczos

function main()

    global temperatures = 0.0:0.1:4

    n = 8
    (hs, spin) = QuantumHamiltonian.Toolkit.spin_system(n, 1//2)

    hamiltonian = NullOperator()
    for i in 1:n
        j = mod(i, n) + 1
        for μ in [:x, :y, :z]
            hamiltonian += spin(i, μ) * spin(j, μ)
        end
    end
    hamiltonian = simplify(hamiltonian)

    hsr = represent(hs)
    hamiltonian_rep = represent(hsr, hamiltonian)

    R = 1000
    D = dimension(hsr)
    krylovdim = min(D, 50)
    zs = zeros(Float64, (R, length(temperatures)))
    Es = zeros(Float64, (R, length(temperatures)))
    @showprogress for r in 1:R
        v = rand(ComplexF64, D) * 2 .- (1 + im)
        normalize!(v)

        iterator = LanczosIterator(x -> hamiltonian_rep*x, v)
        factorization = initialize(iterator)
        for _ in 1:krylovdim
            expand!(iterator, factorization)
            if 
        end
        sm = premeasurestatic(factorization)
        for (it, t) in enumerate(temperatures)
            (z, E, o) = measure(sm, t)
            zs[r, it] = z
            Es[r, it] = E
        end
    end

    global Eavgs = vcat(collect(transpose(sum(Es, dims=1) ./ sum(zs, dims=1)))...)
    h = Matrix(hamiltonian_rep)
    global Eavgs_true = [    minimum(eigvals(Hermitian(h)))    ]
    append!(Eavgs_true, tr(h * exp(-Hermitian(h ./ t))) / tr(exp(-Hermitian(h ./ t)))  for t in temperatures[2:end])
    plot(temperatures, Eavgs, label="FTLM")
    plot!(temperatures, Eavgs_true, label="Direct")
    return Eavgs
end

main()
# Random Matrix Try 1

using FiniteTemperatureLanczos

using LinearAlgebra
using KrylovKit
using Random
using Plots
using ProgressMeter

D = 100
rng = MersenneTwister(0)

m = zeros(ComplexF64, (D, D))
for i in 1:D
    j = mod(i, D) + 1
    v = rand(rng, ComplexF64)
    m[i, j] = v
    m[j, i] = conj(v)
end

o = rand(rng, ComplexF64, (D, D))
# o = o + o'

ts = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
zs = zeros(Float64, length(ts))
as = zeros(ComplexF64, length(ts))
es = zeros(Float64, length(ts))
R = 5000

measured_zs = zeros(Float64, (length(ts), R))
measured_observables = zeros(ComplexF64, (length(ts), R))

@showprogress for r in 1:R
    v = rand(rng, ComplexF64, D) * 2 .- (1 + im)
    normalize!(v)

    iterator = LanczosIterator(m, v)
    factorization = initialize(iterator)
    for _ in 1:100
        expand!(iterator, factorization)
    end

    sm = premeasurestatic(factorization, o, m)
    for (it, t) in enumerate(ts)
        z, energy, obs = measure(sm, t)
        as[it] += obs[2]
        zs[it] += z
        es[it] += energy
        # measured_zs[it, r] = obs[2]
        # measured_observables[it, r] = z
    end
end
as *= (D / R)
zs *= (D / R)

ev = eigvals(m)
true_zs = [sum( exp.(-ev / t)) for t in ts]
true_zs2 = [tr( exp(-m / t)) for t in ts]
true_observables = [
    let ρ = exp(-Hermitian(m/t))
        tr(m * ρ) / tr(ρ)
    end
    for t in ts
]

# plot(ts, real.(true_zs))
# plot(ts, real.(true_zs2))
# plot!(ts, real.(zs))

# plot(ts, real(true_observables))
# plot!(ts, real(as ./ zs))
plot(ts, real(true_observables) .- real(as) ./ zs)

#plot(ts, real(as ./ zs) - real(true_observables))

# plot(ts, imag(true_observables))
# plot!(ts, imag(az ./ zs))


#=

using LinearAlgebra
using KrylovKit

export premeasurestatic
export measure

abstract type AbstractPremeasurement{S<:Number} end

struct BasePremeasurement{R<:Real}
    ρ::Vector{R}
    E::Vector{R}
end

struct StaticObservablePremeasurement{S<:Number} <: AbstractPremeasurement{S}
    A::Matrix{S}
end

struct StaticSusceptibilityPremeasurement{S<:Number} <: AbstractPremeasurement{S}
    A::Array{S, 3}
end


function _prepare_premeasurement(factorization::KrylovKit.LanczosFactorization{T, R}) where {T, R}
    h = SymTridiagonal(factorization.αs, factorization.βs[1:end-1])
    e, u = eigen(h)
    V = basis(factorization).basis
    return (e, u, V)
end


raw"""
Returns (A, E)

where
```math
\begin{aligned}
A_{jl} &= u_{0j} A_{jl} u_{0l}^{r*} \\
E_{jl} &= (\epsilon_j + \epsilon_l)/2
\end{aligned}
```
"""
function premeasurestatic end


# function premeasurestatic(
#     factorization::KrylovKit.LanczosFactorization,
#     observables::AbstractMatrix...
# )
#     return premeasurestatic(factorization, [(x -> m * x) for m in observables]...)
# end

# when observable is callable
function premeasurestatic(
    factorization::KrylovKit.LanczosFactorization{T, R},
    observables...
) where {T, R}
    h = SymTridiagonal(factorization.αs, factorization.βs[1:end-1])
    e, u = eigen(h)

    V = basis(factorization).basis
    d = length(V)
    D = length(V[1])

    A = Matrix[]
    for obs in observables
        a1 = obs(V[1])
        Q = eltype(a1)
        apq = zeros(Q, (d, d))
        for p in 1:d
            apq[p, 1] = dot(V[p], a1)
        end
        for q in 2:d
            aq = obs(V[q])
            for p in 1:d
                apq[p, q] = dot(V[p], aq)
            end
        end
        aij = adjoint(u) * apq * u
        for i in 1:d
            aij[i, :] .*= u[1, i]
        end
        for j in 1:d
            aij[:, j] .*= conj(u[1, j])
        end
        push!(A, aij)
    end
    ρ = abs2.(u[1, :])
    return PremeasurementStatic{R}(A, ρ, e)
end


function measure(sm::PremeasurementStatic{R}, temperature::Real; tol::Real=Base.rtoldefault(R)) where {R}
    if abs(temperature) < tol
        return measurezerotemperature(sm; tol=tol)
    end
    d = length(sm.E)
    half_boltzmann = exp.(-(0.5/temperature) .* (sm.E .- minimum(sm.E)))
    observables = [
        sum(half_boltzmann[i] * half_boltzmann[j] * A[i, j] for i in 1:d for j in 1:d)
        for A in sm.A
    ]
    energy = sum(half_boltzmann[i] * half_boltzmann[i] * sm.ρ[i] * sm.E[i] for i in 1:d)
    energysquared = sum(half_boltzmann[i] * half_boltzmann[i] * sm.ρ[i] * sm.E[i]^2 for i in 1:d)
    partition = sum(half_boltzmann[i] * half_boltzmann[i] * sm.ρ[i] for i in 1:d)
    return (partition=partition, energy=energy, energysquared=energysquared, observables=observables)
end


function measurezerotemperature(sm::PremeasurementStatic{R}; tol::Real=Base.rtoldefault(R)) where {R}
    d = length(sm.E)
    idx_groundstate = findall( <(tol), abs.(sm.E .- minimum(sm.E)))
    observables = [
        sum(A[i, j] for i in idx_groundstate for j in idx_groundstate)
        for A in sm.A
    ]
    energy = sum(sm.ρ[i] * sm.E[i] for i in idx_groundstate)
    energysquared = sum(sm.ρ[i] * sm.E[i]^2 for i in idx_groundstate)
    partition = sum(sm.ρ[i] for i in idx_groundstate)
    return (partition=partition, energy=energy, energysquared=energysquared, observables=observables)
end

=#
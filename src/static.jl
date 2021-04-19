using LinearAlgebra
using KrylovKit

export premeasurestatic
export measure

struct PremeasurementStatic{S<:Number, R<:Real}
    A::Matrix{S}
    ρ::Vector{R}
    E::Vector{R}
end


struct PremeasurementStatic{S<:Tuple{Vararg{<:AbstractMatrix{<:Number}}}, R<:Real}
    A::S
    ρ::Vector{R}
    E::Vector{R}
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


function premeasurestatic(observable::AbstractMatrix, factorization::KrylovKit.LanczosFactorization{T, S}) where {T, S}
    return premeasurestatic(x -> observable * x, factorization)
end

# when observable is callable
function premeasurestatic(observable, factorization::KrylovKit.LanczosFactorization{T, S}) where {T, S}
    reducedhamiltonian = SymTridiagonal(factorization.αs, factorization.βs[1:end-1])
    eigenvalues, reducedeigenvectors = eigen(reducedhamiltonian)
    V = basis(factorization).basis
    d = length(V)
    D = length(V[1])

    Q = eltype(observable(V[1]))
    apq = zeros(Q, (d, d))
    for q in 1:d
        aq = observable(V[q])
        for p in 1:d
            apq[p, q] = dot(V[p], aq)
        end
    end
    aij = adjoint(reducedeigenvectors) * apq * reducedeigenvectors
    for i in 1:d
        aij[i, :] .*= reducedeigenvectors[1, i]
    end
    for j in 1:d
        aij[:, j] .*= conj(reducedeigenvectors[1, j])
    end

    ρi = abs2.(reducedeigenvectors[1, :])
    return PremeasurementStatic{Q, S}(aij, ρi, eigenvalues)
end


function measure(sm::PremeasurementStatic{S, R}, temperature::Real) where {S, R}
    d = length(sm.E)
    half_boltzmann = exp.(-(0.5/temperature) .* sm.E)
    observable = sum(half_boltzmann[i] * half_boltzmann[j] * sm.A[i, j] for i in 1:d for j in 1:d)
    partition = sum(half_boltzmann[i] * half_boltzmann[i] * sm.ρ[i] for i in 1:d)
    return (observable, partition)
end
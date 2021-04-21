using LinearAlgebra
using KrylovKit

export AbstractPremeasurement
export BasePremeasurement

export premeasure
export measure

abstract type AbstractPremeasurement{S<:Number} end

struct BasePremeasurement{
        S<:Number, R<:Real, B<:Number,
        MS<:AbstractMatrix{S},
        MR<:AbstractVector{R},
        MB<:KrylovKit.OrthonormalBasis{<:AbstractVector{B}}
    }
    eigen::Eigen{S, R, MS, MR}
    basis::MB
    function BasePremeasurement(eigen::Eigen{S, R, MS, MR}, basis::MB) where {
            S<:Number, R<:Real, B<:Number,
            MS<:AbstractMatrix{S},
            MR<:AbstractVector{R},
            MB<:KrylovKit.OrthonormalBasis{<:AbstractVector{B}}
        }
        return new{S, R, B, MS, MR, MB}(eigen, basis)
    end
end


function premeasure(factorization::KrylovKit.LanczosFactorization{T, R}) where {T, R}
    h = SymTridiagonal(factorization.αs, factorization.βs[1:end-1])
    return BasePremeasurement(eigen(h), basis(factorization))
end


# aij = ( u ⋅ V† ⋅ A ⋅ V ⋅ u† )
function _project(A::AbstractMatrix, Vb::KrylovKit.OrthonormalBasis, u::AbstractMatrix)
    d = size(A, 1)
    V = Vb.basis
    a1 = A * V[1]
    Q = promote_type(eltype(a1), eltype(u))
    apq = zeros(Q, (d, d))
    for q in 1:d
        aq = A * V[q]
        for p in 1:d
            apq[p, q] = dot(V[p], aq)
        end
    end
    aij = adjoint(u) * apq * u
    return aij
end


function premeasure(pm::BasePremeasurement)
    u = pm.eigen.vectors
    return abs2.(u[1, :])
end


function premeasure(pm::BasePremeasurement, obs::StaticObservable)
    V = pm.basis
    d = length(pm.eigen.values)
    u = pm.eigen.vectors
    A = obs.observable

    aij = _project(A, V, u)
    for i in 1:d
        aij[i, :] .*= u[1, i]
    end
    for j in 1:d
        aij[:, j] .*= conj(u[1, j])
    end
    return aij
end


function premeasure(pm::BasePremeasurement, obs::StaticSusceptibility)
    V = pm.basis
    u = pm.eigen.vectors
    d = length(pm.eigen.values)
    A = obs.observable
    B = obs.field

    aij = _project(A, V, u)
    bjk = _project(B, V, u)
    for i in 1:d
        aij[i, :] .*= u[1, i]
        bjk[:, i] .*= conj(u[1, i])
    end
    Q = promote_type(eltype(aij), eltype(bjk))
    m = zeros(Q, (d, d, d))
    for i in 1:d, j in 1:d, k in 1:d
        m[i, j, k] = aij[i, j] * bjk[j, k]
    end
    return m
end


function measure(pm::BasePremeasurement, temperature::Real; tol::Real=Base.rtoldefault(R)) where {R<:Real}
    E = pm.eigen.values
    if abs(temperature) <= tol
        @warn "it is recommended that you do do not use FTLM for zero temperature property" maxlog=1
        # assume that E contains the ground state
        halfboltzmann = (abs.(E .- minimum(E)) .<= tol)
    else
        # halfboltzmann = exp.( -(0.5/temperature) .* (E .- minimum(E)) )
        halfboltzmann = exp.( -(0.5/temperature) .* E )
    end
    return measure(pm, halfboltzmann)
end


@inline function measure(obs::AbstractArray, E::AbstractVector{R}, temperature::Real; tol::Real=Base.rtoldefault(R)) where {R}
    @boundscheck let
        d = length(E)
        size(obs, 1) == d || throw(DimensionMismatch("$(size(obs)), $d"))
    end
    if abs(temperature) <= tol
        @warn "it is recommended that you do do not use FTLM for zero temperature property" maxlog=1
        # assume that E contains the ground state
        halfboltzmann = (abs.(E .- minimum(E)) .<= tol)
    else
        # halfboltzmann = exp.( -(0.5/temperature) .* (E .- minimum(E)) )
        halfboltzmann = exp.( -(0.5/temperature) .* E )
    end
    return measure(obs, E, halfboltzmann)
end


"""
    Measure partition function
"""
function measure(obs::AbstractVector{<:Number}, E::AbstractVector{<:Real}, halfboltzmann::AbstractVector{<:Real})
    d = length(E)
    @boundscheck let
        d == length(halfboltzmann) || throw(DimensionMismatch("eigenvalues has length $d, half boltzmann has length $(length(halfboltzmann))"))
        size(obs) == (d,) || throw(DimensionMismatch("eigenvalues has length $d, observable has size $(size(obs))"))
    end
    return sum(halfboltzmann[i] * halfboltzmann[i] * obs[i] for i in 1:d)
end


"""
    Measure static observable
"""
function measure(obs::AbstractMatrix{<:Number}, E::AbstractVector{<:Real}, halfboltzmann::AbstractVector{<:Real})
    d = length(E)
    @boundscheck let
        d == length(halfboltzmann) || throw(DimensionMismatch("eigenvalues has length $d, half boltzmann has length $(length(halfboltzmann))"))
        size(obs) == (d, d) || throw(DimensionMismatch("eigenvalues has length $d, observable has size $(size(obs))"))
    end
    return sum(halfboltzmann[i] * halfboltzmann[j] * obs[i, j] for i in 1:d for j in 1:d)
end


"""
    Measure static susceptibility
"""
@inline function measure(obs::AbstractArray{<:Number, 3}, E::AbstractVector{<:Real}, halfboltzmann::AbstractVector{<:Real})
    d = length(E)
    @boundscheck let
        d == length(halfboltzmann) || throw(DimensionMismatch("eigenvalues has length $d, half boltzmann has length $(length(halfboltzmann))"))
        size(obs) == (d, d, d) || throw(DimensionMismatch("eigenvalues has length $d, observable has size $(size(obs))"))
    end
    B = halfboltzmann
    return sum(
        obs[i, j, k] * (B[i] * B[k] - B[j] * B[j]) / (E[j] - 0.5 * (E[i] + E[k]))
        for i in 1:d for j in 1:d for k in 1:d
    )
end


#=
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
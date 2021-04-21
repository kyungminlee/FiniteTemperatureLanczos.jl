using LinearAlgebra
using KrylovKit

export AbstractPremeasurement
export Premeasurement

export premeasure
export measure


"""
    Premeasurement

Contains eigenvalues, eigenvectors, and basis of Krylov factorization.
"""
struct Premeasurement{
        S<:Number, R<:Real, B<:Number,
        MS<:AbstractMatrix{S},
        MR<:AbstractVector{R},
        MB<:KrylovKit.OrthonormalBasis{<:AbstractVector{B}}
    }
    eigen::Eigen{S, R, MS, MR}
    basis::MB
    function Premeasurement(eigen::Eigen{S, R, MS, MR}, basis::MB) where {
            S<:Number, R<:Real, B<:Number,
            MS<:AbstractMatrix{S},
            MR<:AbstractVector{R},
            MB<:KrylovKit.OrthonormalBasis{<:AbstractVector{B}}
        }
        return new{S, R, B, MS, MR, MB}(eigen, basis)
    end
end


"""
    premeasure(factorization::KrylovKit.LanczosFactorization)

Return the Premeasurement object which contains the eigen system and basis of `factorization.`
"""
function premeasure(factorization::KrylovKit.LanczosFactorization{T, R}) where {T, R}
    h = SymTridiagonal(factorization.αs, factorization.βs[1:end-1])
    return Premeasurement(eigen(h), basis(factorization))
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


"""
    premeasure(pm::Premeasurement)

Premeasure partition function. Return the vector elements of the partition function.
"""
premeasure(pm::Premeasurement) = abs2.(pm.eigen.vectors[1, :])


"""
    premeasureenergy(pm::Premeasurement)

Premeasure energy. Return the vector elements of the energy.
"""
premeasureenergy(pm::Premeasurement) = abs2.(pm.eigen.vectors[1, :]) .* pm.eigen.values


"""
    premeasure(pm::Premeasurement, obs::Observable)

Premeasure observable. Return the matrix elements of the observable.
"""
function premeasure(pm::Premeasurement, obs::Observable)
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


"""
    premeasure(pm::Premeasurement, obs::Susceptibility)

Premeasure static susceptibility. Return the tensor elements of susceptibility.
"""
function premeasure(pm::Premeasurement, obs::Susceptibility)
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


"""
    measure(obs, E, temperature)

Measure observable.

# Arguments
- `obs`: tensor elements
- `E`: Ritz eigenvalues
- `temperature`: temperature
"""
function measure(obs::AbstractArray, E::AbstractVector{R}, temperature::Real; tol::Real=Base.rtoldefault(R)) where {R}
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
    measure(obs, E, halfboltzmann)

Measure diagonal observable, including partition function and energy.

# Arguments
- `obs`: tensor elements. A tensor of rank 1, 2, or 3 supported.
- `E`: Ritz eigenvalues
- `halfboltzmann`: exp(-E/2T)
"""
function measure(obs::AbstractVector{<:Number}, E::AbstractVector{<:Real}, halfboltzmann::AbstractVector{<:Real})
    d = length(E)
    @boundscheck let
        d == length(halfboltzmann) || throw(DimensionMismatch("eigenvalues has length $d, half boltzmann has length $(length(halfboltzmann))"))
        size(obs) == (d,) || throw(DimensionMismatch("eigenvalues has length $d, observable has size $(size(obs))"))
    end
    return sum(halfboltzmann[i] * halfboltzmann[i] * obs[i] for i in 1:d)
end

function measure(obs::AbstractMatrix{<:Number}, E::AbstractVector{<:Real}, halfboltzmann::AbstractVector{<:Real})
    d = length(E)
    @boundscheck let
        d == length(halfboltzmann) || throw(DimensionMismatch("eigenvalues has length $d, half boltzmann has length $(length(halfboltzmann))"))
        size(obs) == (d, d) || throw(DimensionMismatch("eigenvalues has length $d, observable has size $(size(obs))"))
    end
    return sum(halfboltzmann[i] * halfboltzmann[j] * obs[i, j] for i in 1:d for j in 1:d)
end

function measure(obs::AbstractArray{<:Number, 3}, E::AbstractVector{<:Real}, halfboltzmann::AbstractVector{<:Real})
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

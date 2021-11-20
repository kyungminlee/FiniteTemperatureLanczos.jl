using LinearAlgebra
using KrylovKit

export AbstractPremeasurement
export Premeasurement
# export ObservableMemspace
# export SusceptibilityMemspace

export premeasure, premeasure!
export premeasureenergy, premeasureenergy!
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
        d = length(eigen.values)
        iszero(d) && throw(ArgumentError("Number of eigenvalues cannot be zero"))
        size(eigen.vectors, 1) == d || throw(ArgumentError("number of eigenvalues $d does not match the size of eigenvectors $(size(eigen.vectors, 1))"))
        size(eigen.vectors, 2) == d || throw(ArgumentError("number of eigenvalues $d does not match the number of eigenvectors $(size(eigen.vectors, 2))"))
        length(basis) == d || throw(ArgumentError("number of basis states $(length(basis)) does not match the size of eigenvectors $d"))
        D = length(first(basis))
        D >= d || throw(ArgumentError("size of basis vectors cannot be smaller than the number of eigenvalues"))
        return new{S, R, B, MS, MR, MB}(eigen, basis)
    end
end


"""
    premeasure(factorization::KrylovKit.LanczosFactorization)

Return the Premeasurement object which contains the eigen system and basis of `factorization.`
"""
function premeasure(factorization::KrylovKit.LanczosFactorization{T, R}) where {T, R}
    h = SymTridiagonal(factorization.αs, factorization.βs)
    return Premeasurement(eigen(h), basis(factorization))
end


# aij = ( u ⋅ V† ⋅ A ⋅ V ⋅ u† )
# 2021/07/30. Above comment wrong? Is aij = ( u† ⋅ V† ⋅ A ⋅ V ⋅ u ) ?
# Hamiltonian:
#   h = V† H V = u e u†   (s.t. H = V h V† within Krylov space)
# Observable:
#   a = V† A V = u ae u†   => ae = u† V† A V u
function _project!(
    aij::AbstractMatrix{Q},
    A::AbstractMatrix{S1},                                # (D, D)
    Vb::KrylovKit.OrthonormalBasis{<:AbstractVector{S2}}, # (D, d). d of vectors of length D
    u::AbstractMatrix{S3},                                # (d, d)
    work::AbstractVector{Q},
) where {Q<:Number, S1<:Number, S2<:Number, S3<:Number}
    V = Vb.basis
    d, D = length(V), length(first(V))
    size(A)   == (D, D) || throw(DimensionMismatch("$(size(A)), $D"))
    size(u)   == (d, d) || throw(DimensionMismatch("$(size(u)), $d"))
    size(aij) == (d, d) || throw(DimensionMismatch("$(size(aij)), $d"))
    length(work) < d*d + D && resize!(work, d*d + D)    
    aq = view(work, 1:D)
    apq = reshape(view(work, D+1:D+d*d), d, d)
    for q in 1:d
        @inbounds LinearAlgebra.mul!(aq, A, V[q])
        for p in 1:d
            @inbounds aij[p, q] = dot(V[p], aq)
        end
    end
    mul!(apq, adjoint(u), aij)
    mul!(aij, apq, u)
    return aij
end


# For Hermitian matrix A, ⟨u|A|v⟩ = ⟨v|A|u⟩*
# The resulting projected matrix should also be Hermitian, BUT NOT OF TYPE Hermitian.
function _project!(
    aij::AbstractMatrix{Q},
    A::Hermitian{S1, <:AbstractMatrix{S1}},               # (D, D)
    Vb::KrylovKit.OrthonormalBasis{<:AbstractVector{S2}}, # (D, d)
    u::AbstractMatrix{S3},                                # (d, d)
    work::AbstractVector{Q},
) where {Q<:Number, S1<:Number, S2<:Number, S3<:Number}
    V = Vb.basis
    d, D = length(V), length(first(V))
    length(work) < d*d + D && resize!(work, d*d + D)
    aq = view(work, 1:D)
    apq = reshape(view(work, D+1:D+d*d), d, d)
    @boundscheck size(A) == (D, D) || throw(DimensionMismatch("$(size(A)), $D"))
    @boundscheck size(u) == (d, d) || throw(DimensionMismatch("$(size(u)), $d"))
    @boundscheck size(aij) == (d, d) || throw(DimensionMismatch("$(size(aij)), $d"))
    for q in 1:d
        @inbounds LinearAlgebra.mul!(aq, A, V[q])
        for p in 1:q-1
            @inbounds val = dot(V[p], aq)
            @inbounds aij[p, q] = val
            @inbounds aij[q, p] = conj(val)
        end
        @inbounds aij[q, q] = dot(V[q], aq)
    end
    mul!(apq, adjoint(u), aij)
    mul!(aij, apq, u)
    return aij
end


"""
    premeasure(pm::Premeasurement)

Premeasure partition function. Return the vector elements of the partition function.
"""
premeasure(pm::Premeasurement) = abs2.(view(pm.eigen.vectors, 1, :))
premeasure!(out::AbstractVector, pm::Premeasurement) = map!(abs2, out, view(pm.eigen.vectors, 1, :))

premeasure(pm::Premeasurement, pow::Integer) = map((x,y) -> abs2(x) * y^pow, view(pm.eigen.vectors, 1, :), pm.eigen.values)
premeasure!(out::AbstractVector, pm::Premeasurement, pow::Integer) = map((x,y) -> abs2(x) * y^pow, out, view(pm.eigen.vectors, 1, :), pm.eigen.values)


"""
    premeasureenergy(pm::Premeasurement)

Premeasure energy. Return the vector elements of the energy.
"""
premeasureenergy(pm::Premeasurement, pow::Integer=1) = map((x,y) -> abs2(x) * y^pow, view(pm.eigen.vectors, 1, :), pm.eigen.values)
premeasureenergy!(out::AbstractVector, pm::Premeasurement, pow::Integer=1) = map((x,y) -> abs2(x) * y^pow, out, view(pm.eigen.vectors, 1, :), pm.eigen.values)

"""
    premeasure(pm::Premeasurement, obs::Observable)

Premeasure observable. Return the matrix elements of the observable.
"""
function premeasure(pm::Premeasurement, obs::Observable, work::AbstractVector{Q}) where {Q}
    d = length(pm.eigen.values)
    aij = Matrix{Q}(undef, (d, d))
    premeasure!(aij, pm, obs, work)
    return aij
end


function premeasure!(aij::AbstractMatrix{Q}, pm::Premeasurement, obs::Observable, work::AbstractVector{Q}) where {Q<:Number}
    V = pm.basis
    d = length(pm.eigen.values)
    u = pm.eigen.vectors
    A = obs.observable
    @boundscheck size(aij) == (d, d) || throw(DimensionMismatch("$(size(aij)), $d"))
    _project!(aij, A, V, u, work)
    for i in 1:d
        @inbounds view(aij, i, :) .*= u[1, i]
        @inbounds view(aij, :, i) .*= conj(u[1, i])
    end
    return aij
end


"""
    premeasure(pm::Premeasurement, obs::Susceptibility)

Premeasure static susceptibility. Return the tensor elements of susceptibility.
"""
function premeasure(pm::Premeasurement, obs::Susceptibility, work::AbstractVector{Q}) where {Q}
    d = length(pm.eigen.values)
    m = Array{Q}(undef, (d, d, d))
    premeasure!(m, pm, obs, work)
    return m
end


"""
    premeasure!(m, pm, susc, work)

# Arguments
- `work::AbstractVector`:: should have at least size D + 2*d² (D+d² for projection, and d² for caching one of the results)
"""
function premeasure!(
    m::AbstractArray{Q, 3},
    pm::Premeasurement,
    obs::Susceptibility,
    work::AbstractVector{Q},
) where {Q}
    V = pm.basis
    u = pm.eigen.vectors
    d = length(pm.eigen.values)
    A = obs.observable
    B = obs.field

    size(m) == (d, d, d) || throw(DimensionMismatch())
    length(work) < D + d*d*2 && resize!(work, D + d*d*2)

    aij = reshape(view(work, D+d*d+1:D+d*d*2), d, d)
    _project!(aij, A, V, u, work)
    for i in 1:d
        view(aij, i, :) .*= u[1, i]
    end
    for k in 1:d
        m[:, :, k] = aij
    end

    bjk = reshape(view(work, D+d*d+1:D+d*d*2), d, d)
    _project!(bjk, B, V, u, work)
    for i in 1:d
        view(bjk, :, i) .*= conj(u[1, i])
    end
    for i in 1:d
        m[i, :, :] .*= bjk
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
function measure(obs::AbstractVector{S1}, E::AbstractVector{S2}, halfboltzmann::AbstractVector{S3}) where {S1<:Number, S2<:Real, S3<:Real}
    d = length(E)
    @boundscheck let
        d == length(halfboltzmann) || throw(DimensionMismatch("eigenvalues has length $d, half boltzmann has length $(length(halfboltzmann))"))
        size(obs) == (d,) || throw(DimensionMismatch("eigenvalues has length $d, observable has size $(size(obs))"))
    end
    out = zero(promote_type(S1, S2, S3))
    for i in 1:d
        @inbounds out += halfboltzmann[i] * halfboltzmann[i] * obs[i]
    end
    return out
    # return mapreduce(x->x[1]*x[2]*x[3], +, zip(halfboltzmann, halfboltzmann, obs))
    # return sum(halfboltzmann[i] * halfboltzmann[i] * obs[i] for i in 1:d)
end

function measure(obs::AbstractMatrix{S1}, E::AbstractVector{S2}, halfboltzmann::AbstractVector{S3}) where {S1<:Number, S2<:Real, S3<:Real}
    d = length(E)
    @boundscheck let
        d == length(halfboltzmann) || throw(DimensionMismatch("eigenvalues has length $d, half boltzmann has length $(length(halfboltzmann))"))
        size(obs) == (d, d) || throw(DimensionMismatch("eigenvalues has length $d, observable has size $(size(obs))"))
    end
    out = zero(promote_type(S1, S2, S3))
    for i in 1:d, j in 1:d
        @inbounds out += halfboltzmann[i] * obs[i, j] * halfboltzmann[j]
    end
    return out
    # return sum(halfboltzmann[i] * halfboltzmann[j] * obs[i, j] for i in 1:d for j in 1:d) # TODO: HERE!
    # return dot(halfboltzmann, obs * halfboltzmann)
    # return mapreduce(x->x[1]*x[2]*x[3], +, zip(halfboltzmann, halfboltzmann, obs))
end

function measure(obs::AbstractArray{S1, 3}, E::AbstractVector{S2}, halfboltzmann::AbstractVector{S3}) where {S1<:Number, S2<:Real, S3<:Real}
    d = length(E)
    @boundscheck let
        d == length(halfboltzmann) || throw(DimensionMismatch("eigenvalues has length $d, half boltzmann has length $(length(halfboltzmann))"))
        size(obs) == (d, d, d) || throw(DimensionMismatch("eigenvalues has length $d, observable has size $(size(obs))"))
    end
    B = halfboltzmann
    out = zero(promote_type(S1, S2, S3))
    for i in 1:d, j in 1:d, k in 1:d
        @inbounds out += obs[i, j, k] * (B[i] * B[k] - B[j] * B[j]) / (E[j] - 0.5 * (E[i] + E[k]))
    end
    return out
    # return sum(
    #     obs[i, j, k] * (B[i] * B[k] - B[j] * B[j]) / (E[j] - 0.5 * (E[i] + E[k]))
    #     for i in 1:d for j in 1:d for k in 1:d
    # )
end




#=
using QuantumHamiltonian
# Computing elements of the operator representation is more costly than accessing elements of the vector.
# The gain from vectorization is smaller than the loss due to duplicate applications of the operator representation.
# => This is what I thought, but it actually runs slower than the naive version.
function _project(A::AbstractOperatorRepresentation, Vb::KrylovKit.OrthonormalBasis, u::AbstractMatrix)
    V = Vb.basis
    d = length(V)
    D = length(V[1])
    Q = promote_type(eltype(A), eltype(V[1]), eltype(u))
    apq = zeros(Q, (d, d))
    let nt = Threads.nthreads()
        local_apq_list = zeros(Q, (nt, d, d))
        Threads.@threads for i in 1:D
        #for i in 1:D
                it = Threads.threadid()
            for (j, v) in QuantumHamiltonian.get_row_iterator(A, i)
                if 0 < j <= D
                    for p in 1:d
                        Vpi_v = conj(V[p][i]) * v
                        for q in 1:d
                            local_apq_list[it, p, q] += Vpi_v * V[q][j]
                            # apq[p, q] += Vpi_v * V[q][j]
                            # local_apq_list[it, p, q] += conj(V[p][i]) * v * V[q][j]
                        end
                    end
                end
            end
        end
        # for it in 1:nt
        #     apq += local_apq_list[it, :, :]
        # end
    end
    aij = adjoint(u) * apq * u
    return aij
end
=#



# struct ObservableMemspace{Q<:Number}
#     matrix::Matrix{Q}
#     vector::Vector{Q}
#     function ObservableMemspace(::Type{Q}, d::Integer, D::Integer) where {Q<:Number}
#         return new{Q}(Matrix{Q}(undef, (d,d)), Vector{Q}(undef, D))
#     end
#     function ObservableMemspace(pm::Premeasurement{S, R, B}) where {S, R, B}
#         Q = promote_type(S, R, B)
#         d = length(pm.basis)
#         D = length(first(pm.basis))
#         return new{Q}(Matrix{Q}(undef, (d,d)), Vector{Q}(undef, D))
#     end
# end


# struct SusceptibilityMemspace{Q<:Number}
#     matrix1::Matrix{Q}
#     matrix2::Matrix{Q}
#     function SusceptibilityMemspace(::Type{Q}, d::Integer) where {Q<:Number}
#         return new{Q}(Matrix{Q}(undef, (d, d)), Matrix{Q}(undef, (d, d)))
#     end
#     function SusceptibilityMemspace(pm::Premeasurement{S, R, B}) where {S, R, B}
#         Q = promote_type(S, R, B)
#         d = length(pm.basis)
#         return new{Q}(Matrix{Q}(undef, (d, d)), Matrix{Q}(undef, (d, d)))
#     end    
# end
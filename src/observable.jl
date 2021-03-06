export AbstractObservable
export Observable
export Susceptibility

import LinearAlgebra

abstract type AbstractObservable{S<:Number} end

struct Observable{S<:Number, M<:AbstractMatrix{<:Number}} <: AbstractObservable{S}
    observable::M
    function Observable(obs::M) where {M<:AbstractMatrix{<:Number}} 
        S = eltype(M)
        LinearAlgebra.checksquare(obs)
        return new{S, M}(obs)
    end
end

struct Susceptibility{S<:Number, Mo<:AbstractMatrix{<:Number}, Mf<:AbstractMatrix{<:Number}} <: AbstractObservable{S}
    observable::Mo
    field::Mf
    function Susceptibility(o::Mo, f::Mf, ::Type{Se}=Float64) where {Mo<:AbstractMatrix{<:Number}, Mf<:AbstractMatrix{<:Number}, Se<:Number}
        S = promote_type(eltype(Mo), eltype(Mf))
        if LinearAlgebra.checksquare(o) != LinearAlgebra.checksquare(f)
            throw(DimensionMismatch("observable has dimensions $(size(o)), field has dimensions $(size(f))"))
        end
        return new{S, Mo, Mf}(o, f)
    end
end

Base.eltype(::AbstractObservable{S}) where S = S
Base.eltype(::Type{<:AbstractObservable{S}}) where S = S
Base.size(obs::Observable, i...) = size(obs.observable, i...)
Base.size(obs::Susceptibility, i...) = size(obs.observable, i...)

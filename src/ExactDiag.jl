module ExactDiag

using SparseArrays
using LinearAlgebra
using Nullables

using Compat
using Reexport

@reexport using Bravais

using UniqueVectors
export AbstractUniqueVector, UniqueVector, findfirst!

using StaticArrays
using SortingAlgorithms
using JLD

import Base: length, checkbounds
import Bravais: translateη

abstract type HilbertSpace{StateType} end

statetype(::HilbertSpace{StateType}) where {StateType} = StateType

# FIXME: work with BitVector

"""
    edapply(f[, x::Real])

Return a closure which multiplies application of a function by a given value.
Often used to provide the parameters in a model Hamiltonian.

For simplicity and consistency, `edapply(f)` with no amplitude defaults to
amplitude 1, as a no-operation.
"""
edapply(f, x::Real) = (i, v) -> f(i, x * v)
edapply(f) = f

"""
    getval(v, i)

Looks up a (possibly site-dependent) parameter.
"""
getval(v::Number, i::Integer) = v
getval(v::Vector{<:Number}, i::Integer) = v[i]

function exp_2πiη(η::Rational{Int})::ComplexF64
    d = denominator(η)
    if d == 1
        return one(ComplexF64)
    elseif d == 2
        @assert numerator(η) & 1 == 1 # otherwise the fraction mustn't be in reduced form
        return -one(ComplexF64)
    else
        return cis(2π * η)
    end
end

function operator_matrix(::Type{T}, hs::HilbertSpace, apply_operator, args...) where {T<:Number}
    length(hs.indexer) > 0 || throw(ArgumentError("Indexer must contain at least one (seed) state."))

    rows = Int[]
    cols = Int[]
    vals = T[]

    j = 1
    while j <= length(hs.indexer)
        apply_operator(hs, j, args...) do i, v
            push!(rows, i)
            push!(cols, j)
            push!(vals, v)
        end
        j += 1
    end

    # NOTE: This function is not type stable, as it may return a real or complex matrix.
    return sparse(rows, cols, isreal(vals) ? real(vals) : vals, length(hs.indexer), length(hs.indexer))
end

operator_matrix(hs::HilbertSpace, args...) = operator_matrix(ComplexF64, hs, args...)

function operator_apply(hs::HilbertSpace, vec::AbstractVector, apply_operator, args...)
    s = length(hs.indexer)
    length(vec) == s || throw(DimensionMismatch())
    rv = zeros(ComplexF64, s)
    for (j, amplitude) in enumerate(vec)
        if amplitude != 0
            apply_operator(hs, j, args...) do i, v
                rv[i] += amplitude * v
            end
        end
    end
    return rv
end

expectval(statevector::AbstractVector, observable::AbstractMatrix) =
    dot(statevector, observable * statevector)

expectval(statevectors::AbstractMatrix, observable::AbstractMatrix) =
    [expectval(statevectors[:, i], observable) for i in 1:size(statevectors, 2)]

expectval(statevectors::AbstractVector{<:AbstractVector}, observable::AbstractMatrix) =
    [expectval(statevector, observable) for statevector in statevectors]

expectval(hs::HilbertSpace, vec::AbstractVector, apply_operator, args...) = dot(vec, operator_apply(hs, vec, apply_operator, args...))

struct CheckEigenstateError <: Exception end

eigenstate_badness(hamiltonian::AbstractMatrix{<:Number}, eigenvalue::Real, eigenvector::AbstractVector{<:Number}) =
    Compat.norm(hamiltonian * eigenvector - eigenvalue * eigenvector)

eigenstate_badness(hs::HilbertSpace, apply_hamiltonian, eigenvalue::Real, eigenvector::AbstractVector{<:Number}) =
    Compat.norm(operator_apply(hs, eigenvector, apply_hamiltonian) - eigenvalue * eigenvector)

check_eigenstate(args...; tolerance::Float64=1e-5) =
    abs(eigenstate_badness(args...)) < tolerance || throw(CheckEigenstateError())

struct HilbertSpaceTranslationCache{HilbertSpaceType<:HilbertSpace}
    hs::HilbertSpaceType
    direction::Int
    cache::Vector{Tuple{Int,Rational{Int}}}

    function HilbertSpaceTranslationCache{HilbertSpaceType}(hs, direction) where {HilbertSpaceType}
        ltrc = LatticeTranslationCache(hs.lattice, direction)
        cache = Tuple{Int,Rational{Int}}[]
        @assert length(hs.indexer) > 0
        sizehint!(cache, length(hs.indexer))
        j = 0 # length(cache)
        while j < length(hs.indexer)
            j += 1
            push!(cache, translateη(hs, ltrc, j))
        end
        new(hs, Int(direction), cache)
    end
end

HilbertSpaceTranslationCache(hs::HilbertSpaceType, direction::Integer) where {HilbertSpaceType<:HilbertSpace} =
    HilbertSpaceTranslationCache{HilbertSpaceType}(hs, direction)

translateη(tc::HilbertSpaceTranslationCache, j::Integer) = tc.cache[j]

include("spinhalf.jl")
include("hubbard.jl")
include("bosons.jl")

include("abelian.jl")

include("time-evolution.jl")

include("entropy.jl")

spinflip_symmetry = (spinflipη, 2)
reflection_symmetry = (reflectionη, 2)
particlehole_symmetry = (particleholeη, 2)

export
    HilbertSpace,
    operator_matrix,
    expectval,
    eigenstate_badness,
    CheckEigenstateError,
    check_eigenstate,
    HilbertSpaceTranslationCache,
    statetype,
    seed_state!,
    get_σz,
    get_charge,
    SpinHalfHilbertSpace,
    spin_half_hamiltonian,
    apply_σ,
    apply_σx,
    apply_σy,
    apply_σz,
    apply_σxσx,
    apply_σzσz,
    apply_σxσx_σyσy,
    apply_σpσm,
    apply_σmσp,
    apply_σpσm_σmσp,
    apply_Sx,
    apply_Sy,
    apply_Sz,
    apply_SxSx,
    apply_SzSz,
    apply_SxSx_SySy,
    apply_SpSm,
    apply_SmSp,
    apply_SpSm_SmSp,
    apply_total_spin_operator,
    SpinHalfFullIndexer,
    HubbardHilbertSpace,
    HubbardParameters,
    hubbard_hamiltonian,
    spinflipη,
    BosonHilbertSpace,
    BosonParameters,
    boson_hamiltonian,
    RepresentativeStateTable,
    DiagonalizationSector,
    apply_reduced_hamiltonian,
    construct_reduced_hamiltonian,
    apply_reduced_operator,
    construct_reduced_operator,
    apply_reduced_commuting_operator,
    construct_reduced_commuting_operator,
    construct_reduced_indexer,
    get_full_psi!,
    get_full_psi,
    time_evolve,
    Tracer,
    diagsizes,
    entanglement_entropy,
    spinflip_symmetry,
    reflection_symmetry,
    particlehole_symmetry

end # module

VERSION >= v"0.4.0-dev+6521" && __precompile__()

module ExactDiag

VERSION < v"0.4-" && using Docile

using Bravais
using IndexedArrays
using SortingAlgorithms
using JLD

import Base: length, checkbounds
import Bravais: translateη

abstract HilbertSpace{StateType}

statetype{StateType}(::HilbertSpace{StateType}) = StateType

# FIXME: work with BitVector

@doc doc"""
Multiplies application of a function by a given value.  Often used to provide the parameters in a model Hamiltonian.

For simplicity and consistency, `edapply(f)` with no amplitude defaults to amplitude 1, as a no-operation.
""" ->
edapply(f, x::Real) = (i, v) -> f(i, x * v)
edapply(f) = f

@doc doc"""
Looks up a (possibly site-dependent) parameter.
""" ->
getval(v::Real, i::Integer) = v
getval{T<:Real}(v::Vector{T}, i::Integer) = v[i]

function exp_2πiη(η::Rational{Int})
    d = den(η)
    if d == 1
        return one(Complex128)
    elseif d == 2
        @assert num(η) & 1 == 1 # otherwise the fraction mustn't be in reduced form
        return -one(Complex128)
    else
        return exp(im * (2π * η))
    end
end

function operator_matrix(hs::HilbertSpace, apply_operator, args...)
    length(hs.indexer) > 0 || throw(ArgumentError("Indexer must contain at least one (seed) state."))

    rows = Int[]
    cols = Int[]
    vals = Complex128[]

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

function operator_apply(hs::HilbertSpace, vec::AbstractVector, apply_operator, args...)
    s = length(hs.indexer)
    length(vec) == s || throw(DimensionMismatch())
    rv = zeros(Complex128, s)
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

expectval{T<:AbstractVector}(statevectors::AbstractVector{T}, observable::AbstractMatrix) =
    [expectval(statevector, observable) for statevector in statevectors]

expectval(hs::HilbertSpace, vec::AbstractVector, apply_operator, args...) = dot(vec, operator_apply(hs, vec, apply_operator, args...))

eigenstate_badness{T<:Number,S<:Number}(hamiltonian::AbstractMatrix{T}, eigenvalue::Real, eigenvector::AbstractVector{S}) =
    vecnorm(hamiltonian * eigenvector - eigenvalue * eigenvector)

eigenstate_badness{S<:Number}(hs::HilbertSpace, apply_hamiltonian, eigenvalue::Real, eigenvector::AbstractVector{S}) =
    vecnorm(operator_apply(hs, eigenvector, apply_hamiltonian) - eigenvalue * eigenvector)

check_eigenstate(args...; tolerance=1e-5) = abs(eigenstate_badness(args...)) < tolerance || throw(InexactError())

immutable HilbertSpaceTranslationCache{HilbertSpaceType<:HilbertSpace}
    hs::HilbertSpaceType
    direction::Int
    cache::Vector{Tuple{Int,Rational{Int}}}

    function HilbertSpaceTranslationCache(hs, direction)
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

HilbertSpaceTranslationCache{HilbertSpaceType<:HilbertSpace}(hs::HilbertSpaceType, direction::Integer) = HilbertSpaceTranslationCache{HilbertSpaceType}(hs, direction)

translateη(tc::HilbertSpaceTranslationCache, j::Integer) = tc.cache[j]

include("spinhalf.jl")
include("hubbard.jl")

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
    check_eigenstate,
    HilbertSpaceTranslationCache,
    statetype,
    seed_state!,
    get_charge,
    SpinHalfHilbertSpace,
    spin_half_hamiltonian,
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
    HubbardHilbertSpace,
    hubbard_hamiltonian,
    spinflipη,
    RepresentativeStateTable,
    DiagonalizationSector,
    apply_reduced_hamiltonian,
    construct_reduced_hamiltonian,
    apply_reduced_operator,
    construct_reduced_operator,
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

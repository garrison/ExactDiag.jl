module ExactDiag

VERSION < v"0.4-" && using Docile

using Bravais
using IndexedArrays
using Compat
using SortingAlgorithms

import Bravais: translateη

abstract HilbertSpace

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
    if length(hs.indexer) > 1
        sizehint!(rows, length(hs.indexer))
        sizehint!(cols, length(hs.indexer))
        sizehint!(vals, length(hs.indexer))
    end

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

expectval(eigenvector::AbstractVector, observable) = dot(eigenvector, observable * eigenvector)
expectval(eigenvectors::AbstractArray, observable) = [expectval(eigenvectors[:, i], observable) for i in 1:size(eigenvectors, 2)]

@compat immutable HilbertSpaceTranslationCache{HilbertSpaceType<:HilbertSpace}
    hs::HilbertSpaceType
    direction::Int
    cache::Vector{Tuple{Int,Rational{Int}}}

    function HilbertSpaceTranslationCache(hs, direction)
        ltrc = LatticeTranslationCache(hs.lattice, direction)
        cache = @compat Tuple{Int,Rational{Int}}[]
        sizehint!(cache, length(hs.indexer))
        for j in 1:length(hs.indexer)
            push!(cache, translateη(hs, ltrc, j))
        end
        new(hs, Int(direction), cache)
    end
end

HilbertSpaceTranslationCache{HilbertSpaceType<:HilbertSpace}(hs::HilbertSpaceType, direction::Integer) = HilbertSpaceTranslationCache{HilbertSpaceType}(hs, direction)

translateη(tc::HilbertSpaceTranslationCache, j::Integer) = tc.cache[j]

include("spinhalf.jl")
include("hubbard.jl")

export
    operator_matrix,
    expectval,
    HilbertSpaceTranslationCache,
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
    apply_Sx,
    apply_Sy,
    apply_Sz,
    apply_SxSx,
    apply_SzSz,
    apply_SxSx_SySy,
    HubbardHilbertSpace,
    hubbard_hamiltonian

end # module

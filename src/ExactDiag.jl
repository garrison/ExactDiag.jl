module ExactDiag

VERSION < v"0.4-" && using Docile

using Bravais
using IndexedArrays
using Compat

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

include("spinhalf.jl")

export
    edapply,
    getval,
    operator_matrix,
    seed_state!,
    SpinHalfHilbertSpace,
    SpinHalfHilbertSpaceTranslationCache,
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
    apply_SxSx_SySy

end # module

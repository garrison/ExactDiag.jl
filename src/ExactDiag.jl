module ExactDiag

VERSION < v"0.4-" && using Docile

using Bravais
using IndexedArrays
using Compat

abstract HilbertSpace

# FIXME: work with BitVector

include("spinhalf.jl")

edapply(f, x::Real) = (i, v) -> f(i, x * v)
edapply(f) = f

function operator_matrix(hs::HilbertSpace, apply_operator)
    length(hs.indexer) > 0 || throw(ArgumentError("Indexer must contain at least one (seed) state."))

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if length(hs.indexer) > 1
        sizehint!(rows, length(hs.indexer))
        sizehint!(cols, length(hs.indexer))
        sizehint!(vals, length(hs.indexer))
    end

    j = 1
    while j <= length(hs.indexer)
        apply_operator(hs, j) do i, v
            push!(rows, i)
            push!(cols, j)
            push!(vals, v)
        end
        j += 1
    end

    return sparse(rows, cols, vals, length(hs.indexer), length(hs.indexer))
end

export SpinHalfHilbertSpace,
    SpinHalfHamiltonian,
    seed_state!,
    apply_hamiltonian,
    apply_translation,
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
    edapply,
    operator_matrix

end # module

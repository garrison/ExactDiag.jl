module ExactDiag

VERSION < v"0.4-" && using Docile

using Bravais
using IndexedArrays

abstract HilbertSpace

# FIXME: work with BitVector

include("spinhalf.jl")

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
    apply_SxSx_SySy

end # module

function test_1d_hubbard_hamiltonian(lattice)
    apply_hamiltonian = hubbard_hamiltonian(t=1, U=3, ϵ_total_spin=pi/1000, ϵ_total_isospin=e/1000)
    indexer = IndexedArray{Vector{Int}}()
    hs = HubbardHilbertSpace(lattice, indexer)
    seed_state!(hs, div(length(lattice), 2), div(length(lattice), 2))
    mat = operator_matrix(hs, apply_hamiltonian)
    @test ishermitian(mat)
    zzz = HilbertSpaceTranslationCache(hs, 1)
    for j in 1:length(hs.indexer)
        i, η = translateη(zzz, j)
        # FIXME: test value of η
        @test hs.indexer[i] == my_1d_translate(hs.indexer[j])
        debug && println("$(hs.indexer[j])\t$(hs.indexer[i])\t$η")
    end
    # FIXME: test GS energy
end
test_1d_hubbard_hamiltonian(ChainLattice([8]))

# Test RepresentativeStateTable and DiagonalizationSector
let
    lattice = ChainLattice([8])
    apply_hamiltonian = spin_half_hamiltonian(J1=1)
    indexer = IndexedArray{Vector{Int}}()
    hs = SpinHalfHilbertSpace(lattice, indexer)
    seed_state!(hs, div(length(lattice), 2))
    rst = RepresentativeStateTable(hs, apply_hamiltonian)
    @test rst.state_info_v[findfirst(hs.indexer, [1,1,1,0,1,0,0,0])].representative_index == rst.state_info_v[findfirst(hs.indexer, [0,1,1,1,0,1,0,0])].representative_index
    @test rst.state_info_v[findfirst(hs.indexer, [1,1,1,0,1,0,0,0])].representative_index != rst.state_info_v[findfirst(hs.indexer, [0,1,1,1,0,0,1,0])].representative_index

    full_ham = operator_matrix(hs, apply_hamiltonian)

    # FIXME: test other sectors too, especially other momentum sectors
    diagsect = DiagonalizationSector(rst, 1, 1)
    reduced_ham = construct_reduced_hamiltonian(diagsect)
    evals, evecs = eigs(reduced_ham, which=:SR)
    for i in 1:length(evals)
        eval = evals[i]
        evec = evecs[:,i]
        full_evec = get_full_psi(diagsect, evec)
        # FIXME: move this to a function
        diff = vecnorm(full_ham * full_evec - eval * full_evec)
        @test_approx_eq_eps diff 0 1e-8
    end
    @test_approx_eq evals[1] -3.651093408937176
end

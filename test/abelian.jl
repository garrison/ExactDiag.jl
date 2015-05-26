# Test RepresentativeStateTable and DiagonalizationSector
#
# Also tests entanglement entropy stuff.
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
        eval = real(evals[i])
        evec = evecs[:,i]
        full_evec = get_full_psi(diagsect, evec)

        @test_approx_eq_eps eigenstate_badness(full_ham, eval, full_evec) 0 1e-8
        @test_approx_eq_eps eigenstate_badness(hs, apply_hamiltonian, eval, full_evec) 0 1e-8
        check_eigenstate(full_ham, eval, full_evec)
        @test_throws InexactError check_eigenstate(full_ham, eval + 1, full_evec)

        if i == 1
            L = length(lattice)
            for L_A in 0:div(L, 2)
                ψ = get_full_psi(diagsect, evec)
                ent_cut1 = entanglement_entropy(Tracer(hs, 1:L_A), ψ)
                ent_cut2 = entanglement_entropy(Tracer(hs, 1:L-L_A), ψ)
                # FIXME: test against known results
                @test_approx_eq_eps ent_cut1 ent_cut2 1e-8
            end
        end
    end
    @test_approx_eq evals[1] -3.651093408937176
end

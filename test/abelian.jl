# Test RepresentativeStateTable and DiagonalizationSector
#
# Also tests entanglement entropy stuff.
let
    L = 8
    lattice = ChainLattice([L])
    apply_hamiltonian = spin_half_hamiltonian(J1=1)
    indexer = IndexedArray{Vector{Int}}()
    hs = SpinHalfHilbertSpace(lattice, indexer)
    seed_state!(hs, div(L, 2))
    rst = RepresentativeStateTable(hs, apply_hamiltonian)
    @test rst.state_info_v[findfirst(hs.indexer, [1,1,1,0,1,0,0,0])].representative_index == rst.state_info_v[findfirst(hs.indexer, [0,1,1,1,0,1,0,0])].representative_index
    @test rst.state_info_v[findfirst(hs.indexer, [1,1,1,0,1,0,0,0])].representative_index != rst.state_info_v[findfirst(hs.indexer, [0,1,1,1,0,0,1,0])].representative_index

    full_ham = operator_matrix(hs, apply_hamiltonian)

    processed_length = 0
    for k_idx in eachmomentumindex(lattice)
        diagsect = DiagonalizationSector(rst, 1, k_idx)
        processed_length += length(diagsect)
        reduced_ham = construct_reduced_hamiltonian(diagsect)
        evals, evecs = eigs(reduced_ham, which=:SR)
        for i in 1:length(evals)
            eval = real(evals[i])
            evec = evecs[:,i]
            ψ = get_full_psi(diagsect, evec)

            @test_approx_eq_eps eigenstate_badness(full_ham, eval, ψ) 0 1e-8
            @test_approx_eq_eps eigenstate_badness(hs, apply_hamiltonian, eval, ψ) 0 1e-8
            check_eigenstate(full_ham, eval, ψ)
            @test_throws InexactError check_eigenstate(full_ham, eval + 1, ψ)

            if i == 1
                for L_A in 0:div(L, 2)
                    ent_cut1 = entanglement_entropy(Tracer(hs, 1:L_A), ψ)
                    ent_cut2 = entanglement_entropy(Tracer(hs, 1:L-L_A), ψ)
                    # FIXME: test against known results
                    @test_approx_eq_eps ent_cut1 ent_cut2 1e-8
                end
            end
        end
        if k_idx == 1
            @test evals[1] ≈ -3.651093408937176
        end
    end
    @test processed_length == length(indexer)
end

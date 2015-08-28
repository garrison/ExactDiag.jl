# Test RepresentativeStateTable and DiagonalizationSector with spin flip
# symmetry
#
# Also tests entanglement entropy stuff.
let
    L = 8
    hs = SpinHalfHilbertSpace(ChainLattice([L]))
    seed_state!(hs, div(L, 2))
    apply_hamiltonian = spin_half_hamiltonian(J1=1)
    rst = RepresentativeStateTable(hs, apply_hamiltonian, [spinflip_symmetry])
    @test rst.state_info_v[findfirst(hs.indexer, [1,1,1,0,1,0,0,0])].representative_index == rst.state_info_v[findfirst(hs.indexer, [0,1,1,1,0,1,0,0])].representative_index
    @test rst.state_info_v[findfirst(hs.indexer, [1,1,1,0,1,0,0,0])].representative_index != rst.state_info_v[findfirst(hs.indexer, [0,1,1,1,0,0,1,0])].representative_index

    full_ham = operator_matrix(hs, apply_hamiltonian)

    @test diagsizes(Tracer(hs, 1:4)) == @compat Dict{Int,Int}(1=>2,4=>2,6=>1)

    processed_length = 0
    for k_idx in 1:nmomenta(hs.lattice)
        for spinflip_idx in 0:1
            diagsect = DiagonalizationSector(rst, 1, k_idx, [spinflip_idx])
            processed_length += length(diagsect)
            reduced_ham = full(construct_reduced_hamiltonian(diagsect))
            fact = eigfact(Hermitian((reduced_ham + reduced_ham') / 2))
            evals, evecs = fact[:values], fact[:vectors]
            for i in 1:length(evals)
                eval = evals[i]
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
                        ent_cut1_dm = entanglement_entropy(Tracer(hs, 1:L_A), ψ * ψ')
                        # FIXME: test against known results
                        @test_approx_eq_eps ent_cut1 ent_cut2 1e-8
                        @test_approx_eq_eps ent_cut1 ent_cut1_dm 1e-12
                    end
                end
            end
            if k_idx == 1 && spinflip_idx == 0
                @test_approx_eq evals[1] -3.651093408937176
            end
        end
    end
    @test processed_length == length(hs.indexer)
end
@test ExactDiag.permutation_parity(@compat Tuple{Int,Int}[]) == 0
@test ExactDiag.permutation_parity([(1,1)]) == 0
@test ExactDiag.permutation_parity([(1,1),(2,2)]) == 0
@test ExactDiag.permutation_parity([(2,2),(1,1)]) == 1
@test ExactDiag.permutation_parity([(1,1),(2,2),(3,3)]) == 0
@test ExactDiag.permutation_parity([(3,3),(1,1),(2,2)]) == 0
@test ExactDiag.permutation_parity([(1,1),(3,3),(2,2)]) == 1

function test_1d_hubbard_hamiltonian(lattice)
    apply_hamiltonian = hubbard_hamiltonian(t=1, U=3, ϵ_total_spin=pi/1000, ϵ_total_pseudospin=e/1000)
    hs = HubbardHilbertSpace(lattice)
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

# With abelian symmetries
let L = 4
    hs = HubbardHilbertSpace(ChainLattice([L]))
    seed_state!(hs, div(L, 2), div(L, 2))
    apply_hamiltonian = hubbard_hamiltonian(t=1, U=2)
    rst = RepresentativeStateTable(hs, apply_hamiltonian, [spinflip_symmetry])

    full_ham = operator_matrix(hs, apply_hamiltonian)

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
                        # FIXME: test against known results
                        @test_approx_eq_eps ent_cut1 ent_cut2 1e-8
                    end
                end
            end
            #println("$L\t$(k_idx-1)\t$(1-2*spinflip_idx)\t$(evals[1])")
        end
    end
    @test processed_length == length(hs.indexer)
end

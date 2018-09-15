# Test RepresentativeStateTable and DiagonalizationSector with reflection
# symmetry at k=0 and k=π
let L = 10
    hs = SpinHalfHilbertSpace(ChainLattice([L]))
    seed_state!(hs, N_up=0)
    apply_hamiltonian = spin_half_hamiltonian(J1_z=1, h_x=0.45, h_z=0.87)
    rst = RepresentativeStateTable(hs, apply_hamiltonian, [reflection_symmetry])

    full_ham = operator_matrix(hs, apply_hamiltonian)
    @test length(hs.indexer) == 2^L

    k_values = [0]
    rem(L, 2) == 0 && push!(k_values, div(L, 2)) # momentum π, if available
    for k_idx in k_values .+ 1
        for reflection_idx in 0:1
            diagsect = DiagonalizationSector(rst, 1, k_idx, [reflection_idx])
            reduced_ham = Matrix(construct_reduced_hamiltonian(diagsect))
            fact = eigen(Hermitian((reduced_ham + reduced_ham') / 2))
            evals, evecs = fact.values, fact.vectors
            for i in 1:length(evals)
                eval = evals[i]
                evec = evecs[:,i]
                ψ = get_full_psi(diagsect, evec)
                check_eigenstate(full_ham, eval, ψ)
            end
        end
    end
end

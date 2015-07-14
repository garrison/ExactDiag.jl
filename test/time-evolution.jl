let
    lattice = ChainLattice([8])
    apply_hamiltonian = spin_half_hamiltonian(J1=1)
    indexer = IndexedArray{Vector{Int}}()
    hs = SpinHalfHilbertSpace(lattice, indexer)
    seed_state!(hs, div(length(lattice), 2))
    rst = RepresentativeStateTable(hs, apply_hamiltonian)

    initial_state = zeros(Complex128, length(rst.hs.indexer))
    initial_state[1] = (1 + im) / sqrt(2)
    time_steps = logspace(-1.5, 4, 41)
    push!(time_steps, 0)
    output_states = time_evolve(rst, initial_state, time_steps) do sector_index, momentum_index
        diagsect = DiagonalizationSector(rst, sector_index, momentum_index)
        fact = eigfact(Hermitian(full(construct_reduced_hamiltonian(diagsect))))
        return construct_reduced_indexer(diagsect), fact[:values], fact[:vectors]
    end

    # Test that time-evolved norms are all (essentially) unity
    for t_i in 1:length(time_steps)
        @test_approx_eq norm(output_states[:, t_i]) 1
    end

    # Test that evolving for zero time returns that same wavefunction
    @test_approx_eq initial_state output_states[:, end]
end

mktempdir() do tmpdir
    # Test dataset multiplication
    n = 30
    m = 10
    a = rand(n,n) + im * rand(n,n)

    fn = joinpath(tmpdir, "exactdiag-mul.jld")
    JLD.save(fn, "a", a)

    jldopen(fn) do file
        b = rand(n, m) + im * rand(n, m)
        @test_approx_eq ExactDiag.my_Ac_mul_B(file["a"], b) (a' * b)

        c = Array(Complex128, size(b)...)
        @test_approx_eq ExactDiag.my_A_mul_B!(c, file["a"], b) (a * b)
    end
end

let
    lattice = ChainLattice([8])
    apply_hamiltonian = spin_half_hamiltonian(J1=1)
    indexer = IndexedArray{Vector{Int}}()
    hs = SpinHalfHilbertSpace(lattice, indexer)
    seed_state!(hs, div(length(lattice), 2))
    rst = RepresentativeStateTable(hs, apply_hamiltonian)

    initial_state = zeros(Complex128, length(rst.hs.indexer))
    initial_state[1] = 1 / sqrt(2)
    initial_state[2] = im / sqrt(2)
    time_steps = logspace(-1.5, 4, 41)
    push!(time_steps, 0)

    function calculate_momentum_sector(sector_index, momentum_index)
        diagsect = DiagonalizationSector(rst, sector_index, momentum_index)
        reduced_hamiltonian = full(construct_reduced_hamiltonian(diagsect))
        fact = eigfact(Hermitian((reduced_hamiltonian + reduced_hamiltonian') / 2))
        return construct_reduced_indexer(diagsect), fact[:values], fact[:vectors]
    end

    # Test that the initial state can be either a Vector or Matrix, with the same results
    let eb = ExactDiag.to_energy_basis(calculate_momentum_sector, rst, initial_state)[1]
        @test ExactDiag.to_energy_basis(calculate_momentum_sector, rst, [initial_state initial_state])[1] == [eb eb]
    end

    let output_states = time_evolve(calculate_momentum_sector, rst, initial_state, time_steps)
        # Test that time-evolved norms are all (essentially) unity
        for t_i in 1:length(time_steps)
            @test_approx_eq norm(output_states[:, t_i]) 1
        end

        # Test that evolving for zero time returns that same wavefunction
        @test_approx_eq initial_state output_states[:, end]

        # Test that evolving forward, then backwards, returns the same wavefunction
        reverse_evolved_state = time_evolve(calculate_momentum_sector, rst, output_states[:, end], [-time_steps[end]])
        @test_approx_eq initial_state reverse_evolved_state

        # Test JLD dataset time evolution
        mktempdir() do tmpdir
            fn = joinpath(tmpdir, "exactdiag-evolve.jld")
            jldopen(fn, "w") do file
                # Save each momentum sector
                for momentum_index in 1:nmomenta(lattice)
                    indexer, evals, evecs = calculate_momentum_sector(1, momentum_index)
                    file["indexer_$(momentum_index)"] = indexer
                    file["evals_$(momentum_index)"] = evals
                    file["evecs_$(momentum_index)"] = evecs
                end

                # Perform time evolution using the saved `JldDataset`s
                output_states_jld = time_evolve(rst, initial_state, time_steps) do sector_index, momentum_index
                    @test sector_index == 1
                    indexer = read(file["indexer_$(momentum_index)"])
                    evals = read(file["evals_$(momentum_index)"])
                    evecs = file["evecs_$(momentum_index)"]
                    return indexer, evals, evecs
                end

                # Check the results
                @test_approx_eq output_states output_states_jld
            end
        end
    end
end

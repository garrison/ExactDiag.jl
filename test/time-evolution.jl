mktempdir() do tmpdir
    VERSION >= v"0.7-" && return # XXX TEST TEMPORARILY DISABLED UNTIL JLD IS FIXED
    # Test dataset multiplication
    n = 30
    m = 10
    a = rand(n,n) + im * rand(n,n)

    fn = joinpath(tmpdir, "exactdiag-mul.jld")
    JLD.save(fn, "a", a)

    jldopen(fn) do file
        b = rand(n, m) + im * rand(n, m)
        @test ExactDiag.my_Ac_mul_B(file["a"], b) ≈ a' * b

        c = Array{ComplexF64}(undef, size(b)...)
        @test ExactDiag.my_A_mul_B!(c, file["a"], b) ≈ a * b
    end
end

let
    lattice = ChainLattice([8])
    apply_hamiltonian = spin_half_hamiltonian(J1=1)
    hs = SpinHalfHilbertSpace(lattice)
    seed_state!(hs, div(length(lattice), 2))
    rst = RepresentativeStateTable(hs, apply_hamiltonian)

    initial_state = zeros(ComplexF64, length(rst.hs.indexer))
    initial_state[1] = 1 / sqrt(2)
    initial_state[2] = im / sqrt(2)
    time_steps = logspace(-1.5, 4, 41)
    push!(time_steps, 0)

    function calculate_momentum_sector(func, sector_index, momentum_index)
        diagsect = DiagonalizationSector(rst, sector_index, momentum_index)
        reduced_hamiltonian = Matrix(construct_reduced_hamiltonian(diagsect))
        fact = eigen(Hermitian((reduced_hamiltonian + reduced_hamiltonian') / 2))
        func(construct_reduced_indexer(diagsect), fact.values, fact.vectors)
        nothing
    end

    # Test that the initial state can be either a Vector or Matrix, with the same results
    let eb = ExactDiag.to_energy_basis(calculate_momentum_sector, rst, initial_state)[1]
        @test ExactDiag.to_energy_basis(calculate_momentum_sector, rst, [initial_state initial_state])[1] == [eb eb]
    end

    let output_states = time_evolve(calculate_momentum_sector, rst, initial_state, time_steps)
        # Test that time-evolved norms are all (essentially) unity
        for t_i in 1:length(time_steps)
            @test norm(output_states[:, t_i]) ≈ 1
        end

        # Test that evolving for zero time returns that same wavefunction
        @test initial_state ≈ output_states[:, end]

        # Test that evolving forward, then backwards, returns the same wavefunction
        reverse_evolved_state = time_evolve(calculate_momentum_sector, rst, output_states[:, end], [-time_steps[end]])
        @test initial_state ≈ reverse_evolved_state

        # Test JLD dataset time evolution
        mktempdir() do tmpdir
            VERSION >= v"0.7-" && return # XXX TEST TEMPORARILY DISABLED UNTIL JLD IS FIXED
            fn = joinpath(tmpdir, "exactdiag-evolve.jld")
            jldopen(fn, "w") do file
                # Save each momentum sector
                for momentum_index in eachmomentumindex(lattice)
                    calculate_momentum_sector(1, momentum_index) do indexer, evals, evecs
                        file["indexer_$(momentum_index)"] = indexer
                        file["evals_$(momentum_index)"] = evals
                        file["evecs_$(momentum_index)"] = evecs
                        nothing
                    end
                end

                # Perform time evolution using the saved `JldDataset`s
                output_states_jld = time_evolve(rst, initial_state, time_steps) do func, sector_index, momentum_index
                    @test sector_index == 1
                    indexer = read(file["indexer_$(momentum_index)"])
                    evals = read(file["evals_$(momentum_index)"])
                    evecs = file["evecs_$(momentum_index)"]
                    func(indexer, evals, evecs)
                    nothing
                end

                # Check the results
                @test output_states ≈ output_states_jld
            end
        end

        # Test multiple initial states
        let output_states2 = time_evolve(calculate_momentum_sector, rst, [initial_state (im * initial_state)], time_steps)
            @test output_states2[:, :, 1] ≈ output_states
            @test output_states2[:, :, 2] ≈ im * output_states
        end

        # Test that doing exact diagonalization without regard to momentum sectors
        # would give the same results
        full_ham = operator_matrix(hs, apply_hamiltonian)
        fact = eigen(Hermitian(Matrix(full_ham)))
        ψ_e = Ac_mul_B(fact.vectors, initial_state)
        ψ_t = fact.vectors * (exp.(-im .* fact.values .* transpose(time_steps)) .* ψ_e)
        @test ψ_t ≈ output_states
    end

    # Try evolving a state whose support is only on a subset of momentum sectors
    initial_state = zeros(ComplexF64, length(rst.hs.indexer))
    initial_state[findfirst(isequal([0,1,0,1,0,1,0,1]), hs.indexer)] = 1

    let output_states = time_evolve(calculate_momentum_sector, rst, initial_state, time_steps, k_indices=[1,5])
        # Test the output is of the correct size
        @test size(output_states) == (length(rst.hs.indexer), length(time_steps))

        # Test that time-evolved norms are all (essentially) unity
        for t_i in 1:length(time_steps)
            @test norm(output_states[:, t_i]) ≈ 1
        end

        # Test that evolving for zero time returns that same wavefunction
        @test initial_state ≈ output_states[:, end]

        # Test against standard evolution
        let output_states2 = time_evolve(calculate_momentum_sector, rst, initial_state, time_steps)
            @test output_states2 ≈ output_states
        end
    end

    # Try providing an incomplete subset of momentum sectors
    let
        ψ_e, = ExactDiag.to_energy_basis(calculate_momentum_sector, rst, initial_state, k_indices=[1])

        # Test that the vector is the correct size
        @test length(ψ_e) < length(rst.hs.indexer)

        # Test that the norm is not unity.
        @test norm(ψ_e) ≈ 1/sqrt(2)
    end
end

# NOTE: We assume we only have enough memory to load one momentum
# sector of the diagonalized Hamiltonian into memory at a time.  We
# therefore make two passes over all the momenta: once when
# constructing the original state in the energy basis, and again when
# moving the time evolved states back to the position basis.

function time_evolve(load_momentum_sector::Function, state_table::RepresentativeStateTable, initial_state::Vector, time_steps::Vector{Float64})
    lattice_nmomenta = nmomenta(state_table.hs.lattice)

    if length(initial_state) != length(state_table.hs.indexer)
        throw(ArgumentError("Initial state must match indexer size"))
    end

    ###
    # Transform initial state to energy basis
    ###
    initial_energy_state = Complex128[]
    all_energies = Float64[]
    for sector_index in 1:state_table.sector_count
        for momentum_index in 1:lattice_nmomenta
            reduced_indexer, reduced_energies, reduced_eigenstates = load_momentum_sector(sector_index, momentum_index)
            @assert length(reduced_indexer) == length(reduced_energies) == size(reduced_eigenstates, 1) == size(reduced_eigenstates, 2)
            diagsect = DiagonalizationSector(state_table, sector_index, momentum_index, reduced_indexer)

            # Project onto current momentum basis
            initial_momentum_state = zeros(Complex128, length(diagsect))
            for (i, (reduced_i, alpha)) in enumerate(diagsect.representative_v)
                if reduced_i != 0
                    initial_momentum_state[reduced_i] += initial_state[i] * conj(alpha)
                end
            end

            # Transform to energy basis
            initial_energy_momentum_state = reduced_eigenstates' * initial_momentum_state
            append!(initial_energy_state, initial_energy_momentum_state)
            append!(all_energies, reduced_energies)
        end
    end
    @assert length(initial_energy_state) == length(all_energies) == length(state_table.hs.indexer)

    ###
    # Time evolve and move back to position basis
    ###
    output_states = zeros(Complex128, length(initial_state), length(time_steps))
    offset = 0
    for sector_index in 1:state_table.sector_count
        for momentum_index in 1:lattice_nmomenta
            reduced_indexer, reduced_energies, reduced_eigenstates = load_momentum_sector(sector_index, momentum_index)
            @assert length(reduced_indexer) == length(reduced_energies) == size(reduced_eigenstates, 1) == size(reduced_eigenstates, 2)
            myrange = offset+1 : offset+length(reduced_indexer)
            diagsect = DiagonalizationSector(state_table, sector_index, momentum_index, reduced_indexer)

            for (t_i, t) in enumerate(time_steps)
                # Time evolve
                time_evolved_sector = initial_energy_state[myrange] .* exp(-im * t * all_energies[myrange])

                # Move back to momentum basis
                momentum_state = reduced_eigenstates * time_evolved_sector

                # Move back to position basis
                for (i, reduced_i, alpha) in diagsect.coefficient_v
                    output_states[i, t_i] += momentum_state[reduced_i] * alpha
                end
            end

            offset += length(reduced_indexer)
        end
    end
    @assert(offset == length(initial_state))

    return output_states
end

# NOTE: We assume we only have enough memory to load one momentum
# sector of the diagonalized Hamiltonian into memory at a time.  We
# therefore make two passes over all the momenta: once when
# constructing the original state in the energy basis, and again when
# moving the time evolved states back to the position basis.

function to_energy_basis(load_momentum_sector::Function, state_table::RepresentativeStateTable, initial_states::@compat(Union{Vector,Matrix}))
    basis_size = length(state_table.hs.indexer)
    if size(initial_states, 1) != basis_size
        throw(ArgumentError("Initial state must match indexer size"))
    end
    nstates = size(initial_states, 2)

    ###
    # Transform initial state to energy basis
    ###
    initial_energy_states = Array(Complex128, size(initial_states)...)
    all_energies = sizehint!(Float64[], basis_size)
    offset = 0
    for sector_index in 1:state_table.sector_count
        for momentum_index in 1:nmomenta(state_table.hs.lattice)
            reduced_indexer, reduced_energies, reduced_eigenstates = load_momentum_sector(sector_index, momentum_index)
            @assert length(reduced_indexer) == length(reduced_energies) == size(reduced_eigenstates, 1) == size(reduced_eigenstates, 2)
            myrange = offset+1 : offset+length(reduced_indexer)
            diagsect = DiagonalizationSector(state_table, sector_index, momentum_index, reduced_indexer)

            initial_momentum_state = Array(Complex128, length(diagsect))

            for x in 1:nstates
                # Project onto current momentum basis
                fill!(initial_momentum_state, 0)
                for (i, (reduced_i, alpha)) in enumerate(diagsect.representative_v)
                    if reduced_i != 0
                        initial_momentum_state[reduced_i] += initial_states[i,x] * conj(alpha)
                    end
                end

                # Transform to energy basis
                initial_energy_states[myrange,x] = Ac_mul_B(reduced_eigenstates, initial_momentum_state)
            end

            append!(all_energies, reduced_energies)

            offset += length(reduced_indexer)
            @assert length(all_energies) == offset
        end
    end
    @assert offset == basis_size

    return initial_energy_states, all_energies
end

function time_evolve_to_position_basis{TimeType<:Real}(load_momentum_sector::Function, state_table::RepresentativeStateTable, initial_energy_state::Vector, time_steps::AbstractVector{TimeType})
    basis_size = length(state_table.hs.indexer)
    if size(initial_energy_state, 1) != basis_size
        throw(ArgumentError("Initial energy state must match indexer size"))
    end

    ###
    # Time evolve and move back to position basis
    ###
    output_states = zeros(Complex128, basis_size, length(time_steps))
    offset = 0
    for sector_index in 1:state_table.sector_count
        for momentum_index in 1:nmomenta(state_table.hs.lattice)
            reduced_indexer, reduced_energies, reduced_eigenstates = load_momentum_sector(sector_index, momentum_index)
            @assert length(reduced_indexer) == length(reduced_energies) == size(reduced_eigenstates, 1) == size(reduced_eigenstates, 2)
            myrange = offset+1 : offset+length(reduced_indexer)
            diagsect = DiagonalizationSector(state_table, sector_index, momentum_index, reduced_indexer)

            for (t_i, t) in enumerate(time_steps)
                # Time evolve
                time_evolved_sector = initial_energy_state[myrange] .* exp(-im * t * reduced_energies)

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
    @assert offset == basis_size

    return output_states
end

function time_evolve(load_momentum_sector::Function, state_table::RepresentativeStateTable, initial_state::Vector, time_steps::AbstractVector{Float64})
    ψ_e, = to_energy_basis(load_momentum_sector, state_table, initial_state)
    return time_evolve_to_position_basis(load_momentum_sector, state_table, ψ_e, time_steps)
end

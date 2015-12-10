# NOTE: We want to load only (part of) one momentum sector of the diagonalized
# Hamiltonian into memory at a time.  We therefore make two passes over all the
# momenta: once when constructing the original state in the energy basis, and
# again when moving the time evolved states back to the position basis.
#
# (FIXME: Actually, it is almost certainly possible to do time evolution using
# just a single pass over the momentum eigenstates, though this may involve
# giving up some flexibility.  Also, for systems that have time-reversal
# invariant and invariant under k <-> -k, we should be able to load eigenstates
# corresponding to only half the Brillouin zone.)

# Implementation of matrix multiplication for JldDataset.  We load one column
# at a time since the matrix is stored in column-major form.  See discussion at
# https://groups.google.com/d/msg/julia-users/cEHmqsMLX5c/NXREyAsVBwAJ

function my_Ac_mul_B(a::JLD.JldDataset, b::Matrix)
    K = size(b, 2)
    M = size(a, 2)
    N = size(a, 1)
    N == size(b, 1) || throw(DimensionMismatch())
    rv = zeros(eltype(b), M, K) # XXX: would be better to use promote_type. see https://github.com/JuliaLang/JLD.jl/issues/28
    for j in 1:M
        col = a[:,j]
        for k in 1:K
            @simd for i in 1:N
                # NOTE: col[i,1] is necessary due to
                # https://github.com/JuliaLang/HDF5.jl/issues/267
                @inbounds rv[j,k] += conj(col[i,1]) * b[i,k]
            end
        end
    end
    return rv
end

function my_A_mul_B!(c::Matrix, a::JLD.JldDataset, b::Matrix)
    @assert c !== b
    K = size(b, 2)
    M = size(a, 1)
    N = size(a, 2)
    size(b, 1) == N || throw(DimensionMismatch())
    size(c) == (M, K) || throw(DimensionMismatch())
    fill!(c, zero(eltype(c)))
    for j in 1:N
        col = a[:,j]
        for k in 1:K
            @simd for i in 1:M
                # NOTE: col[i,1] is necessary due to
                # https://github.com/JuliaLang/HDF5.jl/issues/267
                @inbounds c[i,k] += b[j,k] * col[i,1]
            end
        end
    end
    return c
end

# These functions should behave as usual for all other data types (e.g. normal
# vectors/matrices)
my_Ac_mul_B(a, b) = Ac_mul_B(a, b)
my_A_mul_B!(c, a, b) = A_mul_B!(c, a, b)

function to_energy_basis(load_momentum_sector::Function, state_table::RepresentativeStateTable, initial_states::VecOrMat; k_indices=1:nmomenta(state_table.hs.lattice))
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
    initial_momentum_state = Complex128[]
    for sector_index in 1:state_table.sector_count
        for momentum_index in k_indices
            reduced_indexer, reduced_energies, reduced_eigenstates = load_momentum_sector(sector_index, momentum_index)
            @assert length(reduced_indexer) == length(reduced_energies) == size(reduced_eigenstates, 1) == size(reduced_eigenstates, 2)
            myrange = offset+1 : offset+length(reduced_indexer)
            diagsect = DiagonalizationSector(state_table, sector_index, momentum_index, reduced_indexer)

            initial_momentum_states = zeros(Complex128, length(diagsect), nstates)

            # Project each state onto current momentum basis
            #
            # FIXME: actually, i should make a function in abelian.jl that does
            # this (currently apply_reduced_hamiltonian is the only similar
            # thing).  if so, make sure it fills with zeros first.
            for x in 1:nstates
                for (i, (reduced_i, alpha)) in enumerate(diagsect.representative_v)
                    if reduced_i != 0
                        initial_momentum_states[reduced_i, x] += initial_states[i, x] * conj(alpha)
                    end
                end
            end

            # Transform to energy basis
            initial_energy_states[myrange,:] = my_Ac_mul_B(reduced_eigenstates, initial_momentum_states)

            append!(all_energies, reduced_energies)

            offset += length(reduced_indexer)
            @assert length(all_energies) == offset
        end
    end
    @assert offset == basis_size

    return initial_energy_states, all_energies
end

function time_evolve_to_position_basis{TimeType<:Real}(load_momentum_sector::Function, state_table::RepresentativeStateTable, initial_energy_states::VecOrMat, time_steps::AbstractVector{TimeType}; k_indices=1:nmomenta(state_table.hs.lattice))
    basis_size = length(state_table.hs.indexer)
    if size(initial_energy_states, 1) != basis_size
        throw(ArgumentError("Initial energy state must match indexer size"))
    end

    ###
    # Time evolve and move back to position basis
    ###
    output_states = zeros(Complex128, size(initial_energy_states, 1), length(time_steps), size(initial_energy_states)[2:end]...)
    offset = 0
    for sector_index in 1:state_table.sector_count
        for momentum_index in k_indices
            reduced_indexer, reduced_energies, reduced_eigenstates = load_momentum_sector(sector_index, momentum_index)
            @assert length(reduced_indexer) == length(reduced_energies) == size(reduced_eigenstates, 1) == size(reduced_eigenstates, 2)
            diagsect = DiagonalizationSector(state_table, sector_index, momentum_index, reduced_indexer)

            momentum_states = Array(Complex128, length(diagsect), length(time_steps))
            time_evolved_sector = Array(Complex128, length(diagsect), length(time_steps))

            # Loop through each initial state
            for z in 1:size(initial_energy_states, 2)
                for (t_i, t) in enumerate(time_steps)
                    # Time evolve
                    for i in 1:length(reduced_indexer)
                        time_evolved_sector[i, t_i] = initial_energy_states[offset + i, z] * exp(-im * reduced_energies[i] * t)
                    end
                end

                # Move back to momentum basis
                #
                # FIXME: currently this means loading the reduced_eigenstates
                # once for each initial state.  It would be great to be able to
                # do it once entirely.
                my_A_mul_B!(momentum_states, reduced_eigenstates, time_evolved_sector)

                for (t_i, t) in enumerate(time_steps)
                    # Move back to position basis
                    for (reduced_i, i, alpha) in diagsect.coefficient_v
                        output_states[i, t_i, z] += momentum_states[reduced_i, t_i] * alpha
                    end
                end
            end

            offset += length(reduced_indexer)
        end
    end
    @assert offset == basis_size

    return output_states
end

function time_evolve(load_momentum_sector::Function, state_table::RepresentativeStateTable, initial_states::VecOrMat, time_steps::AbstractVector{Float64}; kwargs...)
    ψ_e, = to_energy_basis(load_momentum_sector, state_table, initial_states; kwargs...)
    return time_evolve_to_position_basis(load_momentum_sector, state_table, ψ_e, time_steps; kwargs...)
end

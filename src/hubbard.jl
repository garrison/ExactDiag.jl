immutable HubbardHilbertSpace{LatticeType<:AbstractSiteNetwork,IndexType<:AbstractIndexedArray} <: HilbertSpace
    lattice::LatticeType
    indexer::IndexType
end

# -- 0
# u- 1
# -d 2
# ud 3

# only the first two bits are allowed to be set
is_valid_hubbard_site_state(site_state::Integer) = (site_state & ~(zero(site_state) $ 3)) == 0

function get_σz(::HubbardHilbertSpace, site_state::Integer)
    @assert is_valid_hubbard_site_state(site_state)
    # maps [0, 1, 2, 3] to [0, 1, -1, 0]
    return (site_state & 1) - ((site_state & 2) >> 1)
end

#"""
#Returns charge on site
#"""
function get_charge(::HubbardHilbertSpace, site_state::Integer)
    @assert is_valid_hubbard_site_state(site_state)
    # maps [0, 1, 2, 3] to [0, 1, 1, 2]
    return (site_state & 1) + ((site_state & 2) >> 1)
end

function get_total_charge(::HubbardHilbertSpace, state::Vector)
    rv = 0
    for x in state
        rv += get_charge(x)
    end
    return rv
end

# In our second quantization, we use the convention that all up creation
# operators come before (or after) the dn creation operators.

# Returns the phase tracker for both spin up and spin dn in the
# respective bits of the return value, under the condition that a
# particle is hopping between sites x1 and x2.
function phase_tracker(state::Vector, x1::Integer, x2::Integer)
    # re-order such that x1 <= x2
    if x1 > x2
        x1, x2 = x2, x1
    end
    @assert x1 <= x2

    rv = 0
    for i in x1+1:x2-1
        rv $= state[i]
    end
    return rv
end

function apply_total_spin_operator(f, hs::HubbardHilbertSpace, s_j::Integer)
    state = hs.indexer[s_j]
    diagonal = 0.0

    for x in 1:length(hs.lattice)
        for x_r in 1:length(hs.lattice)
            # 0.5 * (S^+_i S^-_j + S^-_i S^+_j)
            if x == x_r
                if state[x] == 1 || state[x] == 2
                    diagonal += 0.5
                end
            elseif (state[x] == 1 && state[x_r] == 2) || (state[x] == 2 && state[x_r] == 1)
                other = copy(state)
                other[x], other[x_r] = other[x_r], other[x]
                s_i = findfirst!(hs.indexer, other)

                # figure out whether we need to pick up a phase for the fermions
                pt = phase_tracker(state, x, x_r)
                # the following line maps [0, 1, 2, 3] to [1, -1, -1, 1]
                mult = 1 - ((pt & 2) $ ((pt & 1) << 1))
                @assert mult == 1 || mult == -1

                # the minus sign comes from working through the site hops carefully.
                f(s_i, -0.5 * mult)
            end
            # S^z_i S^z_j
            diagonal += 0.25 * get_σz(hs, state[x]) * get_σz(hs, state[x_r])
        end
    end

    f(s_j, diagonal)
    nothing
end

function apply_total_isospin_operator(f, hs::HubbardHilbertSpace, s_j::Integer)
    @assert isbipartite(hs.lattice)

    state = hs.indexer[s_j]
    diagonal = 0.0

    for x in 1:length(hs.lattice)
        sublattice_index_x = sublattice_index(hs.lattice, x)
        for x_r in 1:length(hs.lattice)
            # 0.5 * (S^+_i S^-_j + S^-_i S^+_j) under duality
            if (x == x_r)
                if state[x] == 0 || state[x] == 3
                    diagonal += 0.5
                end
            elseif (state[x] == 0 && state[x_r] == 3) || (state[x] == 3 && state[x_r] == 0)
                other = copy(state)
                other[x], other[x_r] = other[x_r], other[x]
                s_i = findfirst!(hs.indexer, other)

                # figure out whether we need to pick up a phase for the fermions
                pt = phase_tracker(state, x, x_r)
                # the following line maps [0, 1, 2, 3] to [1, -1, -1, 1]
                mult = 1 - ((pt & 2) $ ((pt & 1) << 1))
                @assert mult == 1 || mult == -1

                sublattice_parity = sublattice_index_x $ sublattice_index(hs.lattice, x_r)
                @assert sublattice_parity & 1 == sublattice_parity
                sublattice_factor = 1 - 2 * sublattice_parity
                f(s_i, 0.5 * sublattice_factor * mult)
            end
            # S^z_i S^z_j under duality
            diagonal += 0.25 * (get_charge(hs, state[x]) - 1) * (get_charge(hs, state[x_r]) - 1)
        end
    end

    f(s_j, diagonal)
    nothing
end

function hubbard_hamiltonian(;
                             t::Real=0.0,
                             U::Real=0.0,
                             V::Real=0.0,
                             W::Real=0.0,
                             J::Real=0.0,
                             J_xy::Real=0.0,
                             J_z::Real=0.0,
                             ϵ_total_spin::Real=0.0,
                             ϵ_total_isospin::Real=0.0)
    if J != 0
        J_xy == J_z == 0 || throw(ArgumentError("If J is provided, J_xy and J_z must not be."))
        J_xy = J_z = J
    end

    return function apply_hamiltonian(f, hs::HubbardHilbertSpace, s_j::Integer)
        state = hs.indexer[s_j]
        diagonal = 0.0

        neighborsη(hs.lattice) do x::Int, x_r::Int, η::Rational{Int}
            # figure out whether we need to pick up a phase
            pt = phase_tracker(state, x, x_r)
            x_r_up_phase = 1 - ((pt & 1) << 1)
            x_r_dn_phase = 1 - (pt & 2)
            e_iθ = exp_2πiη(η)
            @assert x_r_up_phase == 1 || x_r_up_phase == -1
            @assert x_r_dn_phase == 1 || x_r_dn_phase == -1

            # up spin hopping
            const b_up = 1
            if state[x_r] & b_up != 0
                # backward
                if state[x] & b_up == 0
                    other = copy(state)
                    other[x_r] -= b_up
                    other[x] += b_up
                    s_i = findfirst!(hs.indexer, other)
                    f(s_i, -t * x_r_up_phase * e_iθ)
                end
            end
            if state[x] & b_up != 0
                # forward
                if state[x_r] & b_up == 0
                    other = copy(state)
                    other[x] -= b_up
                    other[x_r] += b_up
                    s_i = findfirst!(hs.indexer, other)
                    f(s_i, -t * x_r_up_phase * conj(e_iθ))
                end
            end

            # dn spin hopping
            const b_dn = 2
            if state[x_r] & b_dn != 0
                # backward
                if state[x] & b_dn == 0
                    other = copy(state)
                    other[x_r] -= b_dn
                    other[x] += b_dn
                    s_i = findfirst!(hs.indexer, other)
                    f(s_i, -t * x_r_dn_phase * e_iθ)
                end
            end
            if state[x] & b_dn != 0
                # forward
                if state[x_r] & b_dn == 0
                    other = copy(state)
                    other[x] -= b_dn
                    other[x_r] += b_dn
                    s_i = findfirst!(hs.indexer, other)
                    f(s_i, -t * x_r_dn_phase * conj(e_iθ))
                end
            end

            # Nearest neighbor repulsion "V"
            if V != 0
                c = get_charge(hs, state[x]) * get_charge(hs, state[x_r])
                diagonal += 0.25 * V * c
            end

            # Nearest neighbor doublon repulsion "W"
            if W != 0
                if (state[x] == 3 && state[x_r] == 3)
                    diagonal += 0.25 * W
                end
            end

            # exchange J
            if J_xy != 0
                # 0.5 * (S^+_i S^-_j + S^-_i S^+_j)
                if ((state[x] == 1 && state[x_r] == 2)
                    || (state[x] == 2 && state[x_r] == 1))
                    other = copy(state)
                    other[x], other[x_r] = other[x_r], other[x]
                    s_i = findfirst!(hs.indexer, other)
                    # the minus sign comes from working through the site hops carefully.
                    f(s_i, -0.5 * J_xy * x_r_up_phase * x_r_dn_phase)
                end
            end
            if J_z != 0
                # S^z_i S^z_j
                diagonal += 0.25 * J_z * get_sigma_z(state[x]) * get_sigma_z(state[x_r])
            end
        end

        # Hubbard U
        if U != 0
            doubly_occupied_sites = 0
            for x in 1:length(hs.lattice)
                if state[x] == 3
                    doubly_occupied_sites += 1
                end
            end
            diagonal += U * doubly_occupied_sites
        end

        # total spin term
        if ϵ_total_spin != 0
            apply_total_spin_operator(edapply(f, ϵ_total_spin), hs, s_j)
        end

        # total isospin term
        if ϵ_total_isospin != 0
            apply_total_isospin_operator(edapply(f, ϵ_total_isospin), hs, s_j)
        end

        f(s_j, diagonal)
        nothing
    end
end

function seed_state!(hs::HubbardHilbertSpace, N_up::Integer, N_dn::Integer)
    if !(0 <= N_up <= length(hs.lattice))
        throw(ArgumentError("Invalid N_up provided for size $(length(hs.lattice)) lattice: $(N_up)"))
    end
    if !(0 <= N_dn <= length(hs.lattice))
        throw(ArgumentError("Invalid N_dn provided for size $(length(hs.lattice)) lattice: $(N_dn)"))
    end
    state = zeros(Int, length(hs.lattice))
    for i in 1:N_up
        state[i] $= 1
    end
    for i in 1:N_dn
        state[i] $= 2
    end
    findfirst!(hs.indexer, state)
    return state
end

function translateη(hs::HubbardHilbertSpace, ltrc::LatticeTranslationCache, j::Integer)
    state = hs.indexer[j]
    sz = length(state)
    newstate = zeros(state)
    phase = 0//1

    parity = 0
    for bit in 1:2
        # The pair is the new site index and the creation operator's original
        # position in the array (useful so we can determine the parity of the
        # permutation).
        cdagger = @compat Tuple{Int,Int}[]

        for i in 1:sz
            if state[i] & bit != 0
                new_site_index, η = translateη(ltrc, i)
                @assert 0 < new_site_index <= sz
                newstate[new_site_index] $= bit
                push!(cdagger, (new_site_index, length(cdagger)+1))
                phase -= η
            end
        end

        # Determine parity of permutation.
        sort!(cdagger, alg=TimSort)
        nparticles = length(cdagger)
        inspected = falses(nparticles)
        for i in 1:nparticles
            inspected[i] && continue
            j = i
            while true
                j = cdagger[j][2]
                parity $= 1
                @assert !inspected[j]
                inspected[j] = true
                j == i && break
            end
            parity $= 1
        end
    end

    @assert parity == 0 || parity == 1
    if parity != 0
        phase += 1//2
    end

    return findfirst(hs.indexer, newstate), phase
end
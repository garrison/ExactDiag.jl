# Because resize! will not do the initialization of new elements for us
function my_grow!{T,F}(f::F, vec::AbstractVector{T}, newlen::Integer)
    diff = newlen - length(vec)
    diff == 0 && return vec
    @assert diff > 0
    sizehint!(vec, newlen)
    for i in 1:diff
        push!(vec, f())
    end
    @assert length(vec) == newlen
    return vec
end

my_grow!{T}(vec::AbstractVector{T}, newlen::Integer, value::T=zero(T)) =
    my_grow!(() -> value, vec, newlen)

@compat type StateInfo
    # Currently unused, except in tests.  It can be useful to have
    # this to be able to test if two states are in the same
    # equivalence class, though.  In the future we may be able to
    # optimize some things by knowing this.
    #
    # This gets initialized to 0, which denotes uninitialized.  The
    # RepresentativeStateTable constructor should set it to its final
    # value.
    representative_index::Int

    # If the Hamiltonian contains multiple sectors that do not mix
    # even the Hamiltonian and translation operators are applied, they
    # will be available separately here.  Usually there will only be
    # one sector, unless our basis is mistakenly too large.
    #
    # This gets initialized to 0, which denotes uninitialized.  The
    # RepresentativeStateTable constructor should set it to its final
    # value.
    sector_index::Int

    # The results of translating in each direction.
    #
    # FIXME: Shouldn't we just use a hilbertspace translation cache?
    # One annoying thing is that we might need to generate our states
    # before using it, but that should be fine.
    translation_results::Vector{Tuple{Int, Rational{Int}}}

    StateInfo() = new(0, 0, Tuple{Int, Rational{Int}}[])
end

# It's the job of this class to know all the representative states (and
# what sector each is in) as well as what state is obtained by translating
# each state in each direction.
#
# We may eventually decide to make special optimized versions of this
# class for certain models.
#
# By default (if given no translation_period), it assumes we can translate
# in every direction.
immutable RepresentativeStateTable{HilbertSpaceType<:HilbertSpace}
    # The Hilbert space and the function that applies the Hamiltonian
    #
    # FIXME: Instead of the following two objects separately, maybe we
    # should have an object that represents HilbertSpace +
    # Hamiltonian.
    hs::HilbertSpaceType
    apply_hamiltonian::Function

    # How many times we should apply the translatation operator to
    # advance to the next state.  This is a vector of length `d`.  For
    # open boundary conditions in a direction, set to 0.  For
    # periodic, set to 1.
    #
    # Currently this is assumed to have elements that are all either
    # zero or one.  In the future, we could make it so that e.g. if it
    # has value 2 in a given direction, then it is translated by two
    # sites, not one, when implementing the symmetry.  This could be
    # useful for a Hamiltonian where alternating sites have different
    # properties (and has an even number of sites with PBC).
    translation_period::Vector{Int}

    # StateInfo for each state in hs.indexer
    state_info_v::Vector{StateInfo}

    # A list of indices of each representative state
    representative_state_indices::Vector{Int}

    # Total number of sectors (see above in StateInfo for explanation)
    sector_count::Int

    function RepresentativeStateTable(hs::HilbertSpaceType, apply_hamiltonian::Function,
                                      translation_period::Vector{Int}=Int[])
        d = ndimensions(hs.lattice)

        # FIXME: move this logic to the outer constructor
        if isempty(translation_period)
            # translation by a single step allowed in each direction that
            # is not OBC
            translation_period = [@compat Int(repeater(hs.lattice)[i,i] != 0) for i in 1:d]
        end

        @assert length(translation_period) == d

        @assert length(hs.indexer) > 0 # otherwise we have no seeded states!

        for i in 1:d
            @assert repeater(hs.lattice)[i, i] != 0 || translation_period[i] == 0
        end

        state_info_v = StateInfo[]
        representative_state_indices = Int[]

        CacheType = LatticeTranslationCache{typeof(hs.lattice)}
        ltrc = [translation_period[i] != 0 ? Nullable{CacheType}(LatticeTranslationCache(hs.lattice, i)) : Nullable{CacheType}() for i in 1:d]

        sector_count = 0
        for z in 1:length(hs.indexer)
            if z <= length(state_info_v) && state_info_v[z].sector_index != 0
                # We've already visited this
                continue
            end

            # We've discovered a new sector!
            sector_count += 1
            hamiltonian_basis_queue = Set{Int}()
            push!(hamiltonian_basis_queue, z)

            while !isempty(hamiltonian_basis_queue)
                y = pop!(hamiltonian_basis_queue)

                if y <= length(state_info_v) && state_info_v[y].sector_index != 0
                    # We've already visited this
                    continue
                end

                # We found a new representative state in this sector!
                push!(representative_state_indices, y)

                # Now try to translate in each direction as long as we
                # are finding new states
                translation_basis_queue = Set{Int}()
                push!(translation_basis_queue, y)

                while !isempty(translation_basis_queue)
                    x = pop!(translation_basis_queue)

                    # We should not have already visited this
                    @assert !(x <= length(state_info_v) && state_info_v[x].sector_index != 0)

                    # Make sure our array is big enough, in case the indexer grew
                    my_grow!(StateInfo, state_info_v, length(hs.indexer))
                    @assert 0 < x <= length(state_info_v)

                    state_info_v[x].representative_index = y
                    state_info_v[x].sector_index = sector_count
                    resize!(state_info_v[x].translation_results, d) # NOTE: elements are uninitialized here!

                    # translate in each direction
                    for i in 1:d
                        if translation_period[i] == 0
                            state_info_v[x].translation_results[i] = (x, 0//1)
                        else
                            w = x
                            η = 0//1
                            for j in 1:translation_period[i] # is always one, for now.
                                w, η_inc = translateη(hs, get(ltrc[i]), w)  # FIXME: use a HilbertSpaceTranslationCache instead.  but to do so it must be updateable.  and to be updateable it needs to store the ltrc.  i suppose this makes sense.
                                η += η_inc
                            end
                            if w > length(state_info_v) || state_info_v[w].sector_index == 0
                                push!(translation_basis_queue, w)
                            end
                            state_info_v[x].translation_results[i] = (w, η)
                        end
                    end
                end

                # Now apply the hamiltonian to the current representative
                # state to generate other states
                #
                # NOTE: if we wanted to strictly check translation
                # invariance we would apply the hamiltonian to all states,
                # not just the representative ones.  so in the end we
                # really need to check that it is an eigenstate.
                apply_hamiltonian(hs, y) do newidx, amplitude
                    if amplitude != 0
                        if newidx > length(state_info_v) || state_info_v[newidx].sector_index == 0
                            push!(hamiltonian_basis_queue, newidx)
                        end
                    end
                end
            end

        end

        # FIXME: If being careful, we might as well check that
        # representative_state_indices includes precisely those states
        # that are the representative of themselves, and no repeats.
        # And while we are at it, check that no sector_index or
        # representative_index is 0 anymore.

        return new(hs, apply_hamiltonian, translation_period, state_info_v, representative_state_indices, sector_count)
    end
end

RepresentativeStateTable{HilbertSpaceType<:HilbertSpace}(hs::HilbertSpaceType, apply_hamiltonian::Function) =
    RepresentativeStateTable{HilbertSpaceType}(hs, apply_hamiltonian)

# Takes a RepresentativeStateTable, a momentum, and a sector index.
#
# Builds up normalization for each representative state, an
# IndexedArray with all such states of nonzero norm in our sector, and
# a mapping from each such non-representative state to a
# representative state index (with phase).
immutable DiagonalizationSector{HilbertSpaceType<:HilbertSpace}
    # NOTE: All fields remains constant after initialization.

    state_table::RepresentativeStateTable{HilbertSpaceType}
    sector_index::Int
    momentum_index::Int

    # Includes only nonzero-normed representative states, and refers to states
    # by their index in the main indexer.
    reduced_indexer::IndexedArray{Int}

    # Contains the (nonzero) norm of each element in reduced_indexer
    norm_v::Vector{Float64}

    # Allows us to get \ket{r} and its coefficient from some element of \ket{r_k}
    representative_map::Dict{Int, @compat Tuple{Int, Complex128}}

    # Allows us to get \ket{r_k} from \ket{r}
    coefficient_v::Vector{@compat Tuple{Int, Int, Complex128}}

    function DiagonalizationSector(state_table::RepresentativeStateTable{HilbertSpaceType},
                                   sector_index::Int,
                                   momentum_index::Int,
                                   reduced_indexer::IndexedArray{Int})
        @assert 0 < sector_index <= state_table.sector_count
        @assert 0 < momentum_index <= nmomenta(state_table.hs.lattice)

        norm_v = Float64[]
        representative_map = Dict{Int, @compat Tuple{Int, Complex128}}()
        coefficient_v = @compat Tuple{Int, Int, Complex128}[]

        d = ndimensions(state_table.hs.lattice)
        # This really is just the number of distinct translation steps
        # we may take, including the identity operation. FIXME: rename
        translation_count = 1

        bounds = zeros(Int, d)
        for i in 1:d
            if state_table.translation_period[i] != 0
                @assert state_table.translation_period[i] == 1 # enforce assumption (see XXX below)
                bounds[i] = repeater(state_table.hs.lattice)[i,i]
                translation_count *= bounds[i]
            else
                bounds[i] = 1
            end
        end

        n_discovered_indices = 0

        for z in state_table.representative_state_indices
            if state_table.state_info_v[z].sector_index != sector_index
                continue
            end

            current_terms = Dict{Int, Complex128}()

            # XXX FIXME: make sure all the states in our sector have the same number!
            total_charge = get_total_charge(state_table.hs, z)
            total_momentum = momentum(state_table.hs.lattice, momentum_index, total_charge)

            # FIXME: probably make this its own function with a
            # callback.  then we would need bounds to be a member
            # variable so we are not always generating it.  or heck,
            # let's just pass it as an argument!  no, that doesn't
            # really make sense.
            let
                # NOTE: translation_iteration_helper[d] will *only*
                # ever be translated in the d direction.
                # translation_iteration_helper[d - 1] will be translated
                # both in the d direction and the d-1 direction.
                # translation_iteration_helper[1] is the current element
                # we want.  The other elements of the array are simply
                # stored so we can "wrap around" any time an index is
                # reset to zero (see below).
                translation_iteration_helper = [(z, 0//1) for i in 1:d]
                iter = zeros(Int, d)

                while true
                    # First, do what we need.
                    let
                        # XXX: assumes translation_period contains only zeroes and ones

                        # fixme: we can precalculate each k_dot_r
                        # above and store it in a vector somehow,
                        # though it may be tricky since r is a
                        # "vector".  then again, we can just have a
                        # size_t counter alongside, or use a
                        # multi-dimensional array.
                        kdr = kdotr(state_table.hs.lattice, total_momentum, iter)
                        idx, η = translation_iteration_helper[1]
                        oldval = get!(current_terms, idx, complex(0.0))
                        current_terms[idx] = oldval + exp(complex(0, kdr + 2π * η)) / translation_count
                    end

                    # Second, advance the counter.
                    let
                        i = 1
                        while true
                            i > d && break # we are done iterating; jump out of two loops.
                            iter[i] += 1
                            if iter[i] == bounds[i]
                                iter[i] = 0
                            else
                                break # we are done advancing the counter on this iteration
                            end
                            i += 1
                        end
                        i > d && break

                        # Perform the actual translation associated with
                        # the new counter state.
                        idx, η = translation_iteration_helper[i]
                        idx, η_inc = state_table.state_info_v[idx].translation_results[i]
                        translation_iteration_helper[i] = (idx, η + η_inc)

                        # For each index that wrapped around to zero when
                        # advancing the counter, reset it to be the new
                        # untranslated state
                        for j in 1:i-1
                            translation_iteration_helper[j] = translation_iteration_helper[i]
                        end
                    end
                end
            end

            normsq = 0.0
            for (idx, val) in current_terms
                normsq += abs2(val)
            end
            #println("$(state_table.hs.indexer[z])\t$(momentum_index)\t$(length(current_terms))\t$(normsq)")

            if normsq < 1e-8 # XXX: numerical tuning
                # this state doesn't exist in this momentum sector
                println("skipping!\t", normsq)
                continue
            end

            current_index = findfirst!(reduced_indexer, z)
            n_discovered_indices += 1

            my_grow!(norm_v, length(reduced_indexer))
            norm = sqrt(normsq)
            norm_v[current_index] = norm

            # Save for later some things we have already calculated
            #
            # FIXME: make this optional if we are doing a huge system
            for (idx, val) in current_terms
                # NOTE: if we save these, we probably don't even need
                # norm_v.  in fact, either way it may not make sense
                # to store norm_v.  ugh, looks like we need it after all!
                #
                # XXX: rename these, and write down some algebra equations
                alpha = val / norm;
                representative_map[idx] = (current_index, alpha)
                push!(coefficient_v, (current_index, idx, alpha))
            end
        end

        n_discovered_indices == length(reduced_indexer) || throw(ArgumentError("The provided reduced_indexer contains states that do not exist in this DiagonalizationSector."))

        return new(state_table, sector_index, momentum_index, reduced_indexer, norm_v, representative_map, coefficient_v)
    end
end

DiagonalizationSector{HilbertSpaceType<:HilbertSpace}(state_table::RepresentativeStateTable{HilbertSpaceType}, sector_index::Int, momentum_index::Int) =
    DiagonalizationSector{HilbertSpaceType}(state_table, sector_index, momentum_index, IndexedArray{Int}())

function DiagonalizationSector{HilbertSpaceType<:HilbertSpace}(state_table::RepresentativeStateTable{HilbertSpaceType}, sector_index::Int, momentum_index::Int, provided_reduced_indexer::AbstractVector{HubbardStateType})
    indexer = state_table.indexer
    reduced_indexer = IndexedArray{Int}()
    sizehint!(reduced_indexer, length(provided_reduced_indexer))
    for state in provided_reduced_indexer
        push!(m_reduced_indexer, findfirst!(indexer, state))
    end

    diagsect = DiagonalizationSector{HilbertSpaceType}(state_table, sector_index, momentum_index, reduced_indexer)

    # Make sure we didn't pick up any additional states that weren't
    # in our indexer before
    @assert length(reduced_indexer) == length(provided_reduced_indexer)
    @assert all(diagsect.norm_v .!= 0)

    return diagsect
end

length(diagsect::DiagonalizationSector) = length(diagsect.reduced_indexer)
checkbounds(diagsect::DiagonalizationSector, i::Integer) = 0 < i <= length(diagsect) || throw(BoundsError())

function apply_reduced_hamiltonian(f, diagsect::DiagonalizationSector, reduced_j::Integer)
    checkbounds(diagsect, reduced_j)
    j = diagsect.reduced_indexer[reduced_j]
    reduced_j_norm = diagsect.norm_v[reduced_j]
    diagsect.state_table.apply_hamiltonian(diagsect.state_table.hs, j) do i, amplitude
        # If representative_map does not have the term, it has zero
        # norm and is therefore not in our sector, so we ignore it.
        # This is expected!
        haskey(diagsect.representative_map, i) || return

        reduced_i, phasemult = diagsect.representative_map[i]
        @assert 0 < reduced_i <= length(diagsect)
        f(reduced_i, amplitude * conj(phasemult) / reduced_j_norm)
    end
end

function construct_reduced_hamiltonian(diagsect::DiagonalizationSector)
    s = length(diagsect)
    rows = Int[]
    cols = Int[]
    vals = Complex128[]
    for j in 1:s
        apply_reduced_hamiltonian(diagsect, j) do i, amplitude
            push!(rows, i)
            push!(cols, j)
            push!(vals, amplitude)
        end
    end

    hmat = sparse(rows, cols, vals, s, s)

    # XXX: (numerical) assert hermitian
    #inhermiticity = (hmat - hmat.adjoint()).cwiseAbs().sum();
    #if (inhermiticity > 1e-4) // XXX numerical tuning
    #    std::cerr << "WARNING: matrix is not hermitian by a value of " << inhermiticity << std::endl;
    #assert(inhermiticity < 1); // XXX numerical assert

    # FIXME: possibly return a Hermitian object

    return hmat
end

function construct_reduced_indexer(diagsect::DiagonalizationSector)
    original_indexer = diagsect.state_table.hs.indexer
    return IndexedArray{eltype(original_indexer)}([original_indexer[i] for i in diagsect.reduced_indexer])
end

function get_full_psi(diagsect::DiagonalizationSector, reduced_psi::AbstractVector)
    length(reduced_psi) == length(diagsect) || throw(DimensionMismatch())
    rv = zeros(Complex128, length(diagsect.state_table.hs.indexer))
    for (a, b, c) in diagsect.coefficient_v
        rv[b] = reduced_psi[a] * c
    end
    return rv
end

# FIXME: these diagonalization functions are overkill for systems without symmetry.
#   1) they force the use of ComplexType
#   2) they involve lots of infrastructure and unnecessary build up.
#
# The solution?  For now, simply don't use them unless we have symmetry!

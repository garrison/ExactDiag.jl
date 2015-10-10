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

type StateInfo
    # Currently unused, except in tests.  It can be useful to have
    # this to be able to test if two states are in the same
    # equivalence class, though.  In the future we may be able to
    # optimize some things by knowing this.
    #
    # This gets initialized to 0, which denotes uninitialized.  The
    # RepresentativeStateTable constructor should set it to its final
    # value.
    representative_index::Int

    # If the Hamiltonian contains multiple sectors that do not mix when the
    # Hamiltonian and translation/symmetry operators are applied, they will be
    # available separately here.  Usually there will only be one sector, unless
    # our basis is mistakenly too large.
    #
    # This gets initialized to 0, which denotes uninitialized.  The
    # RepresentativeStateTable constructor should set it to its final
    # value.
    sector_index::Int

    # The results of translating in each direction, as well as results of any
    # additional symmetry operations provided.  The length of this vector is
    # the number of dimensions plus the number of additional symmetries given.
    transformation_results::Vector{@compat Tuple{Int, Rational{Int}}}

    StateInfo() = new(0, 0, @compat(Tuple{Int, Rational{Int}}[]))
end

# It's the job of this class to know all the representative states (and what
# sector each is in) as well as what state is obtained by translating each
# state in each direction (and applying each additional symmetry operation).
#
# We may eventually decide to make special optimized versions of this
# class for certain models.
#
# By default (if given no transformation_exponent_v), it assumes we can
# translate in every direction (except those with open boundary conditions) and
# apply each provided "additional symmetry" operation (e.g. spin flip).
immutable RepresentativeStateTable{HilbertSpaceType<:HilbertSpace}
    # The Hilbert space and the function that applies the Hamiltonian
    #
    # FIXME: Instead of the following two objects separately, maybe we
    # should have an object that represents HilbertSpace +
    # Hamiltonian.
    hs::HilbertSpaceType
    apply_hamiltonian::Function

    # How many times we should apply each translation (or additional symmetry)
    # operator to advance to the next state.  This is a vector of length `dd`
    # (i.e. the number of dimensions plus the number of additional symmetries).
    # For open boundary conditions in a direction, set to 0.  For periodic, set
    # to 1.
    #
    # Currently this is assumed to have elements that are all either
    # zero or one.  In the future, we could make it so that e.g. if it
    # has value 2 in a given direction, then it is translated by two
    # sites, not one, when implementing the symmetry.  This could be
    # useful for a Hamiltonian where alternating sites have different
    # properties (and has an even number of sites with PBC).
    transformation_exponent_v::Vector{Int}

    # StateInfo for each state in hs.indexer
    state_info_v::Vector{StateInfo}

    # A list of indices of each representative state
    representative_state_indices::Vector{Int}

    # Total number of sectors (see above in StateInfo for explanation)
    sector_count::Int

    # The period of each "additional symmetry operation" that was provided
    # (e.g. spin flip has period 2, since two applications brings us back to
    # the original state)
    additional_symmetry_periods::Vector{Int}

    function RepresentativeStateTable(hs::HilbertSpaceType, apply_hamiltonian::Function,
                                      additional_symmetries::Vector{@compat Tuple{Function,Int}}=(@compat Tuple{Function,Int})[],
                                      transformation_exponent_v::Vector{Int}=Int[])
        for (symm_func, symm_period) in additional_symmetries
            @assert symm_period > 0
        end

        d = ndimensions(hs.lattice)
        dd = d + length(additional_symmetries)

        # FIXME: move this logic to the outer constructor
        if isempty(transformation_exponent_v)
            # transformation by a single step allowed in each direction that is
            # not OBC
            transformation_exponent_v = [@compat Int(i > d || repeater(hs.lattice)[i,i] != 0) for i in 1:dd]
        end

        @assert length(transformation_exponent_v) == dd

        @assert length(hs.indexer) > 0 # otherwise we have no seeded states!

        for i in 1:d
            @assert repeater(hs.lattice)[i, i] != 0 || transformation_exponent_v[i] == 0
        end
        # (nothing equivalent to check for d+1:dd)

        for i in 1:dd
            # see XXX below
            @assert transformation_exponent_v[i] == 0 || transformation_exponent_v[i] == 1
        end

        state_info_v = StateInfo[]
        representative_state_indices = Int[]

        CacheType = LatticeTranslationCache{typeof(hs.lattice)}
        ltrc = [transformation_exponent_v[i] != 0 ? Nullable{CacheType}(LatticeTranslationCache(hs.lattice, i)) : Nullable{CacheType}() for i in 1:d]

        transformation_basis_queue = IndexedArray{Int}()
        hamiltonian_basis_queue = IndexedArray{Int}()

        sector_count = 0
        for z in 1:length(hs.indexer)
            if z <= length(state_info_v) && state_info_v[z].representative_index != 0
                # We've already visited this
                continue
            end

            # We've discovered a new sector!
            sector_count += 1
            @assert isempty(hamiltonian_basis_queue)
            push!(hamiltonian_basis_queue, z)

            while !isempty(hamiltonian_basis_queue)
                y = pop!(hamiltonian_basis_queue)

                if y <= length(state_info_v) && state_info_v[y].representative_index != 0
                    # We've already visited this
                    continue
                end

                # We found a new representative state in this sector!
                push!(representative_state_indices, y)

                # Now try to translate in each direction (and perform each
                # additional symmetry operation) as long as we are finding new
                # states
                @assert isempty(transformation_basis_queue)
                push!(transformation_basis_queue, y)

                while !isempty(transformation_basis_queue)
                    x = pop!(transformation_basis_queue)

                    # We should not have already visited this
                    @assert !(x <= length(state_info_v) && state_info_v[x].representative_index != 0)

                    # Make sure our array is big enough, in case the indexer grew
                    my_grow!(StateInfo, state_info_v, length(hs.indexer))
                    @assert 0 < x <= length(state_info_v)

                    state_info_v[x].representative_index = y
                    state_info_v[x].sector_index = sector_count
                    resize!(state_info_v[x].transformation_results, dd) # NOTE: elements are uninitialized here!

                    # Translate in each direction, and perform each additional
                    # symmetry operation on the current state to generate
                    # states in the same class.
                    for i in 1:dd
                        if transformation_exponent_v[i] == 0
                            state_info_v[x].transformation_results[i] = (x, 0//1)
                        else
                            w = x
                            η = 0//1
                            for j in 1:transformation_exponent_v[i] # is always 0 or 1, for now.
                                w, η_inc = (i <= d) ? translateη(hs, get(ltrc[i]), w) : additional_symmetries[i-d][1](hs, w)
                                η += η_inc
                            end
                            if w > length(state_info_v) || state_info_v[w].representative_index == 0
                                findfirst!(transformation_basis_queue, w)
                            end
                            state_info_v[x].transformation_results[i] = (w, η)
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
                        if newidx > length(state_info_v) || state_info_v[newidx].representative_index == 0
                            findfirst!(hamiltonian_basis_queue, newidx)
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

        additional_symmetry_periods = [symm_period for (symm_func, symm_period) in additional_symmetries]
        return new(hs, apply_hamiltonian, transformation_exponent_v, state_info_v, representative_state_indices, sector_count, additional_symmetry_periods)
    end
end

RepresentativeStateTable{HilbertSpaceType<:HilbertSpace}(hs::HilbertSpaceType, apply_hamiltonian::Function, additional_symmetries::Vector{@compat Tuple{Function,Int}}=(@compat Tuple{Function,Int})[]) =
    RepresentativeStateTable{HilbertSpaceType}(hs, apply_hamiltonian, additional_symmetries)

# At times we will want to be able to specify which states are used as the
# representative ones (e.g. if we are loading the results of a previous
# calculation).  This constructor handles this case by immediately modifying
# the representative states once the state table has been constructed.
function RepresentativeStateTable{StateType}(hs::HilbertSpace{StateType}, apply_hamiltonian::Function, representative_states::AbstractVector{StateType}, additional_args...)
    state_table = RepresentativeStateTable(hs, apply_hamiltonian, additional_args...)

    # Now we just need to set the representative states to those given

    # Plan the mutation
    old2new = Dict{Int,Int}() # maps the old representative index to the new one
    for state in representative_states
        newidx = findfirst(state_table.hs.indexer, state)
        oldidx = state_table.state_info_v[newidx].representative_index
        haskey(old2new, oldidx) && throw(ArgumentError("Two states were given in the same equivalence class"))
        old2new[oldidx] = newidx
    end
    for oldidx in state_table.representative_state_indices
        haskey(old2new, oldidx) || throw(ArgumentError("The states provided do not account for all representative states"))
    end

    # Perform the mutation
    for stateinfo in state_table.state_info_v
        stateinfo.representative_index = old2new[stateinfo.representative_index]
    end
    for (i, oldidx) in enumerate(state_table.representative_state_indices)
        state_table.representative_state_indices[i] = old2new[oldidx]
    end

    return state_table
end

function iterate_transformations{F}(f::F, state_table::RepresentativeStateTable, z::Integer, bounds::Vector{Int})
    dd = length(bounds) # i.e. number of dimensions plus number of additional symmetry operations

    # NOTE: iteration_helper[dd] will *only* ever be incremented in the dd
    # direction.  iteration_helper[dd - 1] will be incremented both in the dd
    # direction and the dd-1 direction.  iteration_helper[1] is the current
    # element we want.  The other elements of the array are simply stored so we
    # can "wrap around" any time an index is reset to zero.
    iteration_helper = [(z, 0//1) for i in 1:dd]
    iter = zeros(Int, dd)

    while true
        # First, do what we need.
        f(iter, iteration_helper[1])

        # Second, advance the counter.
        i = 1
        while true
            i > dd && return # we are done iterating
            iter[i] += 1
            iter[i] != bounds[i] && break # we have advanced the counter
            iter[i] = 0 # index i gets wrapped around to zero
            i += 1
        end

        # Perform the actual translation (or symmetry operation) associated
        # with the new counter state.
        idx, η = iteration_helper[i]
        idx, η_inc = state_table.state_info_v[idx].transformation_results[i]
        iteration_helper[i] = (idx, η + η_inc)

        # For each index that wrapped around to zero when
        # advancing the counter, reset it to be the new
        # untransformed state
        for j in 1:i-1
            iteration_helper[j] = iteration_helper[i]
        end
    end
end

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

    # Allows us to get the representative state \ket{r} and its
    # coefficient given a state \ket{s} (some element of \ket{r_k}).
    representative_v::Vector{@compat Tuple{Int, Complex128}}

    # Allows us to explicitly construct \ket{r_k} from \ket{r}
    coefficient_v::Vector{@compat Tuple{Int, Int, Complex128}}

    function DiagonalizationSector(state_table::RepresentativeStateTable{HilbertSpaceType},
                                   sector_index::Int,
                                   momentum_index::Int,
                                   reduced_indexer::IndexedArray{Int},
                                   additional_symmetry_indices::Vector{Int}=Int[])
        @assert 0 < sector_index <= state_table.sector_count
        @assert 0 < momentum_index <= nmomenta(state_table.hs.lattice)

        @assert length(additional_symmetry_indices) == length(state_table.additional_symmetry_periods)
        @assert all(0 .<= additional_symmetry_indices .< state_table.additional_symmetry_periods)

        norm_v = Float64[]
        representative_v = @compat Tuple{Int, Complex128}[(0, 0.0im) for i in 1:length(state_table.hs.indexer)]
        coefficient_v = @compat Tuple{Int, Int, Complex128}[]

        d = ndimensions(state_table.hs.lattice)
        dd = d + length(state_table.additional_symmetry_periods)

        # This will become the number of distinct combinations of
        # transformation (i.e. translation + additional symmetry) steps we may
        # take, including the identity operation.  It is impicitly assumed that
        # all such operations commute with one another.
        transformation_count = 1

        bounds = zeros(Int, dd)
        for i in 1:dd
            if state_table.transformation_exponent_v[i] != 0
                @assert state_table.transformation_exponent_v[i] == 1 # enforce assumption (see XXX below)
                bounds[i] = (i <= d) ? repeater(state_table.hs.lattice)[i,i] : state_table.additional_symmetry_periods[i-d]
                transformation_count *= bounds[i]
            else
                bounds[i] = 1
            end
        end

        additional_symmetry_mult = [x//y for (x, y) in zip(additional_symmetry_indices, state_table.additional_symmetry_periods)]

        n_discovered_indices = 0

        for z in state_table.representative_state_indices
            if state_table.state_info_v[z].sector_index != sector_index
                continue
            end

            current_terms = Dict{Int, Complex128}()

            # XXX FIXME: make sure all the states in our sector have the same number!
            total_charge = get_total_charge(state_table.hs, z)
            total_momentum = momentum(state_table.hs.lattice, momentum_index, total_charge)

            iterate_transformations(state_table, z, bounds) do iter, current_transformation
                # XXX: assumes transformation_exponent_v contains only zeroes and ones

                # fixme: we can precalculate each k_dot_r
                # above and store it in a vector somehow,
                # though it may be tricky since r is a
                # "vector".  then again, we can just have a
                # size_t counter alongside, or use a
                # multi-dimensional array.
                idx, η = current_transformation
                kdr = kdotr(state_table.hs.lattice, total_momentum, iter[1:d])
                η += dot(iter[d+1:dd], additional_symmetry_mult)
                oldval = get!(current_terms, idx, complex(0.0))
                current_terms[idx] = oldval + exp(complex(0, kdr + 2π * η)) / transformation_count
            end

            normsq = mapreduce(abs2, +, values(current_terms))
            #println("$(state_table.hs.indexer[z])\t$(momentum_index)\t$(length(current_terms))\t$(normsq)")

            if normsq < 1e-8 # XXX: numerical tuning
                # this state doesn't exist in this momentum sector
                continue
            end

            reduced_i = findfirst!(reduced_indexer, z)
            n_discovered_indices += 1

            my_grow!(norm_v, length(reduced_indexer))
            norm = sqrt(normsq)
            norm_v[reduced_i] = norm

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
                representative_v[idx] = (reduced_i, alpha)
                push!(coefficient_v, (reduced_i, idx, alpha))
            end
        end

        n_discovered_indices == length(reduced_indexer) || throw(ArgumentError("The provided reduced_indexer contains states that do not exist in this DiagonalizationSector."))

        # This is necessary for us to be able to perform a binary search using
        # searchsorted().
        sort!(coefficient_v)

        return new(state_table, sector_index, momentum_index, reduced_indexer, norm_v, representative_v, coefficient_v)
    end
end

DiagonalizationSector{HilbertSpaceType<:HilbertSpace}(state_table::RepresentativeStateTable{HilbertSpaceType}, sector_index::Int, momentum_index::Int, additional_symmetry_indices::Vector{Int}=Int[]) =
    DiagonalizationSector{HilbertSpaceType}(state_table, sector_index, momentum_index, IndexedArray{Int}(), additional_symmetry_indices)

function DiagonalizationSector{StateType,HilbertSpaceType<:HilbertSpace}(state_table::RepresentativeStateTable{HilbertSpaceType}, sector_index::Int, momentum_index::Int, provided_reduced_indexer::AbstractVector{StateType}, additional_symmetry_indices::Vector{Int}=Int[])
    # because I've been unable to figure out how to require this in the method signature
    @assert StateType == statetype(state_table.hs)

    indexer = state_table.hs.indexer
    reduced_indexer = IndexedArray{Int}()
    #sizehint!(reduced_indexer, length(provided_reduced_indexer))
    for state in provided_reduced_indexer
        i = findfirst!(indexer, state)
        if state_table.state_info_v[i].representative_index != i
            throw(ArgumentError("reduced_indexer contains states that our state_table has not chosen to be representative"))
        end
        push!(reduced_indexer, i)
    end

    diagsect = DiagonalizationSector{HilbertSpaceType}(state_table, sector_index, momentum_index, reduced_indexer, additional_symmetry_indices)

    # Make sure we didn't pick up any additional states that weren't
    # in our indexer before
    @assert length(reduced_indexer) == length(provided_reduced_indexer)
    @assert all(x -> x != 0, diagsect.norm_v)

    return diagsect
end

length(diagsect::DiagonalizationSector) = length(diagsect.reduced_indexer)
checkbounds(diagsect::DiagonalizationSector, i::Integer) = 0 < i <= length(diagsect) || throw(BoundsError())

function apply_reduced_hamiltonian(f, diagsect::DiagonalizationSector, reduced_j::Integer)
    # This, of course, assumes that the Hamiltonian commutes with any
    # translations allowed by the diagsect.
    checkbounds(diagsect, reduced_j)
    j = diagsect.reduced_indexer[reduced_j]
    reduced_j_norm = diagsect.norm_v[reduced_j]
    diagsect.state_table.apply_hamiltonian(diagsect.state_table.hs, j) do i, amplitude
        reduced_i, alpha = diagsect.representative_v[i]
        # If reduced_i is zero, this term has zero norm and is
        # therefore not in our sector, so we ignore it.  This is
        # expected!
        if reduced_i != 0
            @assert 0 < reduced_i <= length(diagsect)
            f(reduced_i, amplitude * conj(alpha) / reduced_j_norm)
        end
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
    #inhermiticity = sum(abs(hmat - hmat'))
    #if inhermiticity > 1e-4 # XXX numerical tuning
    #    println("WARNING: matrix is not hermitian by a value of $(inhermiticity)")
    #end
    #@assert inhermiticity < 1 # XXX numerical assert

    # FIXME: possibly return a Hermitian object once Julia supports sparse,
    # Hermitian matrices

    return hmat
end

function apply_reduced_operator(f, diagsect::DiagonalizationSector, reduced_j::Integer, apply_operator, args...)
    # Unlike apply_reduced_hamiltonian, this does *not* assume that the
    # operator commutes with the translation operators, so is therefore slower
    # to execute.
    checkbounds(diagsect, reduced_j)
    for z in searchsorted(diagsect.coefficient_v, reduced_j, by=(a -> a[1]))
        reduced_j_, j, alpha_j = diagsect.coefficient_v[z]
        apply_operator(diagsect.state_table.hs, j, args...) do i, amplitude
            reduced_i, alpha_i = diagsect.representative_v[i]
            # If reduced_i is zero, this term has zero norm and is
            # therefore not in our sector, so we ignore it.  This is
            # expected!
            if reduced_i != 0
                @assert 0 < reduced_i <= length(diagsect)
                f(reduced_i, amplitude * conj(alpha_i) * alpha_j)
            end
        end
    end
end

function construct_reduced_operator(diagsect::DiagonalizationSector, apply_operator, args...)
    s = length(diagsect)
    rows = Int[]
    cols = Int[]
    vals = Complex128[]
    for j in 1:s
        apply_reduced_operator(diagsect, j, apply_operator, args...) do i, amplitude
            push!(rows, i)
            push!(cols, j)
            push!(vals, amplitude)
        end
    end

    return sparse(rows, cols, vals, s, s)
end

function construct_reduced_indexer(diagsect::DiagonalizationSector)
    original_indexer = diagsect.state_table.hs.indexer
    return IndexedArray{eltype(original_indexer)}([original_indexer[i] for i in diagsect.reduced_indexer])
end

function get_full_psi!(full_psi::Vector{Complex128}, diagsect::DiagonalizationSector, reduced_psi::AbstractVector)
    length(reduced_psi) == length(diagsect) || throw(DimensionMismatch("reduced_psi has the wrong length"))
    length(full_psi) == length(diagsect.state_table.hs.indexer) || throw(DimensionMismatch("full_psi has the wrong length"))
    fill!(full_psi, zero(Complex128))
    for (reduced_i, i, alpha) in diagsect.coefficient_v
        full_psi[i] = reduced_psi[reduced_i] * alpha
    end
    return full_psi
end

get_full_psi(diagsect::DiagonalizationSector, reduced_psi::AbstractVector) =
    get_full_psi!(Array(Complex128, length(diagsect.state_table.hs.indexer)), diagsect, reduced_psi)

# FIXME: these diagonalization functions are overkill for systems without symmetry.
#   1) they force the use of ComplexType
#   2) they involve lots of infrastructure and unnecessary build up.
#
# The solution?  For now, simply don't use them unless we have symmetry!

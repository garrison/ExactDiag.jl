immutable TracerSector{StateType<:AbstractVector}
    # The indexed states in each subregion.  These define what is
    # meant by idx_A and idx_B below.
    indexer_A::IndexedArray{StateType}
    indexer_B::IndexedArray{StateType}

    # Quick lookup of what states are compatible with a given idx_A:
    # by_A[idx_A] contains (idx_B, idx) for each state that looks like
    # idx_A in region A.
    #
    # and vice versa for idx_B:
    # by_B[idx_B] contains a list of (idx_A, idx)
    #
    # NOTE: idx is the index of the full state in the original indexer
    by_A::Vector{Vector{@compat Tuple{Int,Int}}}
    by_B::Vector{Vector{@compat Tuple{Int,Int}}}

    # (idx, idx_A, idx_B) for each state in the sector
    backmap::Vector{@compat Tuple{Int, Int, Int}}

    # To make sure any wavefunctions ψ we are given have the correct
    # number of elements
    original_basis_length::Int
end

immutable Tracer{StateType<:AbstractVector}
    sectors::Vector{TracerSector{StateType}}

    function Tracer(_sites_A::AbstractVector{Int}, nsites::Int, basis)
        all(0 .< _sites_A .<= nsites) || throw(ArgumentError("Subsystem contains sites that do not exist in the system."))

        # This first line fixes things up if we specify a site twice
        # in sites_A, or if it's not ordered.
        sites_A = filter(i -> (i in _sites_A), 1:nsites)
        sites_B = filter(i -> !(i in sites_A), 1:nsites)

        # First figure out and organize all the basis states in A and
        # B, without regard to sector
        preliminary_indexer_A = IndexedArray{StateType}()
        preliminary_indexer_B = IndexedArray{StateType}()
        preliminary_backmap = @compat Tuple{Int, Int}[]
        for state in basis
            idx_A = findfirst!(preliminary_indexer_A, state[sites_A])
            idx_B = findfirst!(preliminary_indexer_B, state[sites_B])
            push!(preliminary_backmap, (idx_A, idx_B))
        end
        preliminary_by_A = [Int[] for i in 1:length(preliminary_indexer_A)]
        preliminary_by_B = [Int[] for i in 1:length(preliminary_indexer_B)]
        for (idx, (idx_A, idx_B)) in enumerate(preliminary_backmap)
            push!(preliminary_by_A[idx_A], idx_B)
            push!(preliminary_by_B[idx_B], idx_A)
        end

        # Now figure out the independent sectors
        sectors = TracerSector{StateType}[]
        remaining_A = Set(1:length(preliminary_indexer_A))
        remaining_B = Set(1:length(preliminary_indexer_B))
        while !isempty(remaining_A)
            @assert !isempty(remaining_B)

            # Find all states in the current sector
            indexer_A = IndexedArray{StateType}()
            indexer_B = IndexedArray{StateType}()
            let
                idx_A_queue = Int[pop!(remaining_A)]
                idx_B_queue = Int[]
                while !(isempty(idx_A_queue) && isempty(idx_B_queue))
                    while !isempty(idx_A_queue)
                        idx_A = pop!(idx_A_queue)
                        @assert idx_A ∉ remaining_A
                        push!(indexer_A, preliminary_indexer_A[idx_A])
                        for idx_B in preliminary_by_A[idx_A]
                            if idx_B in remaining_B
                                delete!(remaining_B, idx_B)
                                push!(idx_B_queue, idx_B)
                            end
                        end
                    end
                    while !isempty(idx_B_queue)
                        idx_B = pop!(idx_B_queue)
                        @assert idx_B ∉ remaining_B
                        push!(indexer_B, preliminary_indexer_B[idx_B])
                        for idx_A in preliminary_by_B[idx_B]
                            if idx_A in remaining_A
                                delete!(remaining_A, idx_A)
                                push!(idx_A_queue, idx_A)
                            end
                        end
                    end
                end
            end

            # Construct by_A, by_B, and backmap
            backmap = @compat Tuple{Int, Int, Int}[]
            for (idx, state) in enumerate(basis)
                if state[sites_A] ∉ indexer_A
                    continue
                end
                idx_A = findfirst(indexer_A, state[sites_A])
                idx_B = findfirst(indexer_B, state[sites_B])
                push!(backmap, (idx, idx_A, idx_B))
            end
            by_A = [@compat Tuple{Int, Int}[] for i in 1:length(indexer_A)]
            by_B = [@compat Tuple{Int, Int}[] for i in 1:length(indexer_B)]
            for (idx, idx_A, idx_B) in backmap
                push!(by_A[idx_A], (idx_B, idx))
                push!(by_B[idx_B], (idx_A, idx))
            end

            push!(sectors, TracerSector(indexer_A, indexer_B, by_A, by_B, backmap, length(basis)))
        end
        @assert isempty(remaining_B)

        return new(sectors)
    end
end

Tracer(hs::HilbertSpace, sites_A) = Tracer{statetype(hs)}(sites_A, length(hs.lattice), hs.indexer)

function diagsizes(tracer::Tracer)
    # Returns the number of matrices of each size that must be diagonalized to
    # calculate the QDL diagnostic of a single state
    rv = Dict{Int,Int}()
    for sector in tracer.sectors
        s = length(sector.indexer_A)
        rv[s] = get(rv, s, 0) + 1
    end
    return rv
end

function construct_ρ_A_block(ts::TracerSector, ψ)
    length(ψ) == ts.original_basis_length || throw(ArgumentError("Wavefunction ψ has the wrong number of elements"))
    ρ_A = zeros(Complex128, length(ts.indexer_A), length(ts.indexer_A))
    for idx_B in 1:length(ts.indexer_B)
        for (a1, i1) in ts.by_B[idx_B]
            for (a2, i2) in ts.by_B[idx_B]
                ρ_A[a1, a2] += ψ[i1] * ψ[i2]'
            end
        end
    end
    # It should be Hermitian by construction (FIXME: right?)
    return Hermitian(ρ_A)
end

function entanglement_entropy{T<:Number}(tracer::Tracer, ψ::AbstractVector{T}, alpha::Real=1)
    if alpha == 1
        rv = 0.0
        for sector in tracer.sectors
            ρ_A = construct_ρ_A_block(sector, ψ)
            for v in eigvals(ρ_A)
                v > 0 || continue
                rv -= v * log(v)
            end
        end
        return rv
    else
        s = 0.0
        for sector in tracer.sectors
            ρ_A = construct_ρ_A_block(sector, ψ)
            for v in eigvals(ρ_A)
                v > 0 || continue
                s += v ^ alpha
            end
        end
        return log(s) / (1 - alpha)
    end
end

function entanglement_entropy{T<:Real}(eigenvalues::AbstractVector{T}, alpha::Real=1)
    if alpha == 1
        rv = 0.0
        for v in eigenvalues
            v > 0 || continue
            rv -= v * log(v)
        end
        return rv
    else
        s = 0.0
        for v in eigenvalues
            v > 0 || continue
            s += v ^ alpha
        end
        return log(s) / (1 - alpha)
    end
end

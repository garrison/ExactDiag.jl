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
    by_A::Vector{Vector{Tuple{Int,Int}}}
    by_B::Vector{Vector{Tuple{Int,Int}}}

    # (idx, idx_A, idx_B) for each state in the sector
    backmap::Vector{Tuple{Int, Int, Int}}

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
        preliminary_backmap = Tuple{Int, Int}[]
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
        remaining_A = IntSet(1:length(preliminary_indexer_A))
        remaining_B = IntSet(1:length(preliminary_indexer_B))
        while !isempty(remaining_A)
            @assert !isempty(remaining_B)

            # Find all states in the current sector
            indexer_A_set = Set{StateType}()
            indexer_B_set = Set{StateType}()
            let
                idx_A_queue = Int[pop!(remaining_A)]
                idx_B_queue = Int[]
                while !(isempty(idx_A_queue) && isempty(idx_B_queue))
                    while !isempty(idx_A_queue)
                        idx_A = pop!(idx_A_queue)
                        @assert idx_A ∉ remaining_A
                        push!(indexer_A_set, preliminary_indexer_A[idx_A])
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
                        push!(indexer_B_set, preliminary_indexer_B[idx_B])
                        for idx_A in preliminary_by_B[idx_B]
                            if idx_A in remaining_A
                                delete!(remaining_A, idx_A)
                                push!(idx_A_queue, idx_A)
                            end
                        end
                    end
                end
            end

            # Sort things to be a bit more predictable
            #
            # XXX: This makes assumptions about StateType that should probably be moved elsewhere.
            indexer_A = IndexedArray{StateType}(sort!(collect(indexer_A_set), by=(x -> (x...))))
            indexer_B = IndexedArray{StateType}(sort!(collect(indexer_B_set), by=(x -> (x...))))

            # Construct by_A, by_B, and backmap
            backmap = Tuple{Int, Int, Int}[]
            for (idx, state) in enumerate(basis)
                if state[sites_A] ∉ indexer_A
                    continue
                end
                idx_A = findfirst(indexer_A, state[sites_A])
                idx_B = findfirst(indexer_B, state[sites_B])
                push!(backmap, (idx, idx_A, idx_B))
            end
            by_A = [Tuple{Int, Int}[] for i in 1:length(indexer_A)]
            by_B = [Tuple{Int, Int}[] for i in 1:length(indexer_B)]
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

function construct_psimat_block!{T<:Number}(psimat::Matrix{T}, ts::TracerSector, ψ::AbstractVector{T})
    length(ψ) == ts.original_basis_length || throw(DimensionMismatch("Wavefunction ψ has the wrong number of elements"))
    size(psimat) == (length(ts.indexer_A), length(ts.indexer_B)) || throw(DimensionMismatch("psimat has the wrong size"))
    fill!(psimat, zero(T))
    for b in 1:length(ts.indexer_B)
        for (a, i) in ts.by_B[b]
            psimat[a, b] = ψ[i]
        end
    end
    return psimat
end

function construct_ρ_A_block!{T<:Number}(ρ_A::AbstractMatrix{T}, psimat::Matrix{T}, ts::TracerSector, ψ::AbstractVector{T})
    size(ρ_A) == (length(ts.indexer_A), length(ts.indexer_A)) || throw(DimensionMismatch("Density matrix ρ_A has the wrong size"))
    construct_psimat_block!(psimat, ts, ψ)
    A_mul_Bc!(ρ_A, psimat, psimat)
    # It should be Hermitian by construction (FIXME: right?)
    #@assert sum(abs(ρ_A - ρ_A')) == 0
    return Hermitian(ρ_A)
end

construct_ρ_A_block!{T<:Number}(ρ_A::AbstractMatrix{T}, ts::TracerSector, ψ::AbstractVector{T}) =
    construct_ρ_A_block!(ρ_A, Array(T, length(ts.indexer_A), length(ts.indexer_B)), ts, ψ)

function construct_ρ_A_block!{T<:Number}(ρ_A::AbstractMatrix{T}, ts::TracerSector, ρ::AbstractMatrix{T})
    size(ρ) == (ts.original_basis_length, ts.original_basis_length) || throw(DimensionMismatch("Density matrix ρ has the wrong size"))
    # FIXME: make sure ρ is Hermitian, or require Hermitian type
    size(ρ_A) == (length(ts.indexer_A), length(ts.indexer_A)) || throw(DimensionMismatch("Density matrix ρ_A has the wrong size"))
    fill!(ρ_A, zero(T))
    for idx_B in 1:length(ts.indexer_B)
        z = ts.by_B[idx_B]
        for (a2, i2) in z
            for (a1, i1) in z
                @inbounds ρ_A[a1, a2] += ρ[i1, i2]
            end
        end
    end
    #@assert sum(abs(ρ_A - ρ_A')) == 0
    return Hermitian(ρ_A)
end

function construct_ρ_A_block{T<:Number}(ts::TracerSector, ψ_or_ρ::AbstractVecOrMat{T})
    M = length(ts.indexer_A)
    return construct_ρ_A_block!(Array(T, M, M), ts, ψ_or_ρ)
end

# Apparently julia does not special-case eigvals() for small, Hermitian
# matrices.  When possible, we return the closed-form solution instead of
# invoking the full eigenvalue machinery.  This can lead to efficiency
# improvements since for some calculations many of our blocks of ρ_A will be
# 1x1 or 2x2.
function myeigvals(mat::Hermitian)
    if size(mat, 1) == 1
        # Trivial
        return [real(mat[1,1])]
    elseif size(mat, 2) == 2
        # Obtain the eigenvalues from the quadratic formula
        a = real(mat[1,1])
        d = real(mat[2,2])
        apd = a + d
        amd = a - d
        desc = sqrt(amd * amd + 4 * abs2(mat[1,2]))
        return [(apd - desc) / 2, (apd + desc) / 2]
    else
        # Call the full eigenvalue machinery
        return eigvals(mat)
    end
end

function entanglement_entropy{T<:Number}(tracer::Tracer, ψ_or_ρ::AbstractVecOrMat{T}, alpha::Real=1)
    if alpha == 1
        rv = 0.0
        for sector in tracer.sectors
            ρ_A = construct_ρ_A_block(sector, ψ_or_ρ)
            for v in myeigvals(ρ_A)
                v > 0 || continue
                rv -= v * log(v)
            end
        end
        return rv
    else
        s = 0.0
        for sector in tracer.sectors
            ρ_A = construct_ρ_A_block(sector, ψ_or_ρ)
            for v in myeigvals(ρ_A)
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

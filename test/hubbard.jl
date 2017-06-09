@test ExactDiag.permutation_parity(Int[]) == 0
@test ExactDiag.permutation_parity([1]) == 0
@test ExactDiag.permutation_parity([1,2]) == 0
@test ExactDiag.permutation_parity([2,1]) == 1
@test ExactDiag.permutation_parity([1,2,3]) == 0
@test ExactDiag.permutation_parity([3,1,2]) == 0
@test ExactDiag.permutation_parity([1,3,2]) == 1

@test ExactDiag.site_spinflip(0) == 0
@test ExactDiag.site_spinflip(1) == 2
@test ExactDiag.site_spinflip(2) == 1
@test ExactDiag.site_spinflip(3) == 3

function test_1d_hubbard_hamiltonian(lattice)
    apply_hamiltonian = hubbard_hamiltonian(t=1, U=3, ϵ_total_spin=pi/1000, ϵ_total_pseudospin=e/1000)
    hs = HubbardHilbertSpace(lattice)
    seed_state!(hs, div(length(lattice), 2), div(length(lattice), 2))
    mat = operator_matrix(hs, apply_hamiltonian)
    @test ishermitian(mat)
    zzz = HilbertSpaceTranslationCache(hs, 1)
    for j in 1:length(hs.indexer)
        i, η = translateη(zzz, j)
        # FIXME: test value of η
        @test hs.indexer[i] == my_1d_translate(hs.indexer[j])
        debug && println("$(hs.indexer[j])\t$(hs.indexer[i])\t$η")
    end
    # FIXME: test GS energy
end
test_1d_hubbard_hamiltonian(ChainLattice([8]))

# With abelian symmetries (this one always assumes half filling)
function test_1d_hubbard_symmetries{F<:Function}(lattice, symmetries::Vector{Tuple{F,Int}})
    L = length(lattice)
    hs = HubbardHilbertSpace(lattice)
    seed_state!(hs, div(L, 2), div(L, 2))
    apply_hamiltonian = hubbard_hamiltonian(t=1, U=2)
    rst = RepresentativeStateTable(hs, apply_hamiltonian, symmetries)

    full_ham = operator_matrix(hs, apply_hamiltonian)

    processed_length = 0
    symmbounds = (repeat([2], inner=[length(symmetries)])...)
    for k_idx in eachmomentumindex(hs.lattice)
        for symm_idx in 1:2*length(symmetries)
            symm = [ind2sub(symmbounds, symm_idx)...] - 1
            diagsect = DiagonalizationSector(rst, 1, k_idx, symm)
            length(diagsect) != 0 || continue
            processed_length += length(diagsect)
            reduced_ham = full(construct_reduced_hamiltonian(diagsect))
            @test vecnorm(reduced_ham - reduced_ham') < 1e-5
            fact = eigfact(Hermitian((reduced_ham + reduced_ham') / 2))
            evals, evecs = fact[:values], fact[:vectors]
            for i in 1:length(evals)
                eval = evals[i]
                evec = evecs[:,i]
                ψ = get_full_psi(diagsect, evec)
                @test eigenstate_badness(full_ham, eval, ψ) ≈ 0 atol=1e-8
            end
        end
    end
    @test processed_length == length(hs.indexer)
end

let
    for L in 2:2:6
        # Take advantage of spinflip symmetry
        test_1d_hubbard_symmetries(ChainLattice([L]), [spinflip_symmetry])
        test_1d_hubbard_symmetries(ChainLattice([L], diagm([L]), [1//2]), [spinflip_symmetry])
        test_1d_hubbard_symmetries(ChainLattice([L], diagm([L]), [1//3]), [spinflip_symmetry])
        test_1d_hubbard_symmetries(ChainLattice([L], diagm([0])), [spinflip_symmetry])

        # Particle-hole symmetry
        test_1d_hubbard_symmetries(ChainLattice([L]), [particlehole_symmetry])
        test_1d_hubbard_symmetries(ChainLattice([L], diagm([L]), [1//2]), [particlehole_symmetry])
        test_1d_hubbard_symmetries(ChainLattice([L], diagm([0])), [particlehole_symmetry])

        # Spinflip + particle-hole
        test_1d_hubbard_symmetries(ChainLattice([L]), [spinflip_symmetry, particlehole_symmetry])
        test_1d_hubbard_symmetries(ChainLattice([L], diagm([L]), [1//2]), [spinflip_symmetry, particlehole_symmetry])
        test_1d_hubbard_symmetries(ChainLattice([L], diagm([0])), [spinflip_symmetry, particlehole_symmetry])
    end
end

let
    lattice = ChainLattice([6], diagm([6]), [1//7])
    ltrc = LatticeTranslationCache(lattice, 1)
    hs = HubbardHilbertSpace(lattice)
    push!(hs.indexer, [0, 0, 3, 1, 0, 2])
    t1, η1 = translateη(hs, ltrc, 1)
    @test t1 == findfirst(hs.indexer, [2, 0, 0, 3, 1, 0]) == 2
    @test η1 == 5//14
    t2, η2 = translateη(hs, ltrc, 2)
    @test t2 == findfirst(hs.indexer, [0, 2, 0, 0, 3, 1]) == 3
    @test η2 == 0//1
    t3, η3 = translateη(hs, ltrc, 3)
    @test t3 == findfirst(hs.indexer, [1, 0, 2, 0, 0, 3]) == 4
    @test η3 == 5//14
    t4, η4 = translateη(hs, ltrc, 4)
    @test t4 == findfirst(hs.indexer, [3, 1, 0, 2, 0, 0]) == 5
    @test η4 == -2//7
    t5, η5 = translateη(hs, ltrc, 5)
    @test t5 == findfirst(hs.indexer, [0, 3, 1, 0, 2, 0]) == 6
    @test η5 == 0//1
    t6, η6 = translateη(hs, ltrc, 6)
    @test t6 == 1
    @test η6 == 0//1
end

# explicit Slater determinant for ground state
let
    L = 6
    lattice = ChainLattice([6])
    apply_hamiltonian = hubbard_hamiltonian(t=1)
    hs = HubbardHilbertSpace(lattice)
    seed_state!(hs, div(L, 2), div(L, 2))

    rst = RepresentativeStateTable(hs, apply_hamiltonian, [spinflip_symmetry])

    # Find the ground state wavefunction
    local gs_eval, gs_evec
    gs_eval = maxintfloat(Float64) # fixme
    for k_idx in eachmomentumindex(hs.lattice)
        for spinflip_idx in 0:1
            diagsect = DiagonalizationSector(rst, 1, k_idx, [spinflip_idx])
            length(diagsect) != 0 || continue
            reduced_ham = full(construct_reduced_hamiltonian(diagsect))
            @test vecnorm(reduced_ham - reduced_ham') < 1e-5
            fact = eigfact(Hermitian((reduced_ham + reduced_ham') / 2))
            evals, evecs = fact[:values], fact[:vectors]
            eval = evals[1]
            evec = evecs[:,1]
            ψ = get_full_psi(diagsect, evec)
            if eval < gs_eval
                gs_eval = eval
                gs_evec = ψ
            end
        end
    end
    debug && @show gs_eval

    # Construct the Slater determinant wavefunction
    slater = Array{Complex128}(length(hs.indexer))
    for (i, state) in enumerate(hs.indexer)
        # First figure out the positions of the particles
        pn = 3
        r_up = find(state) do x
            x & 1 != 0
        end
        @assert length(r_up) == pn
        r_dn = find(state) do x
            x & 2 != 0
        end
        @assert length(r_dn) == pn

        # Now construct and calculate the two determinants.
        mat1 = Array{Complex128}(pn, pn)
        mat2 = Array{Complex128}(pn, pn)
        for j in 1:3
            mat1[j, 1] = exp(im * pi * r_up[j] / 3.)
            mat1[j, 2] = 1
            mat1[j, 3] = exp(-im * pi * r_up[j] / 3.)
            mat2[j, 1] = exp(im * pi * r_dn[j] / 3.)
            mat2[j, 2] = 1
            mat2[j, 3] = exp(-im * pi * r_dn[j] / 3.)
        end

        # Normalize properly
        mat1 /= sqrt(6)
        mat2 /= sqrt(6)

        slater[i] = det(mat1) * det(mat2)
    end

    overlap = dot(gs_evec, slater)
    debug && @show overlap
    @test abs(overlap) ≈ 1
end

# Slater determinants for all eigenstates
function degenerate_ranges{T<:Real}(v::AbstractVector{T}, tol::T)
    @assert issorted(v)
    diffs = v[2:end] - v[1:end-1]
    indices = find(diffs) do d
        d > tol
    end
    return UnitRange{Int}[x:y for (x, y) in zip([1; indices + 1], [indices; length(v)])]
end

@test degenerate_ranges([1,2,3,4.0], 0.5) == UnitRange{Int}[1:1, 2:2, 3:3, 4:4]
@test degenerate_ranges([1,2,3,4.0], 1.2) == UnitRange{Int}[1:4]
@test degenerate_ranges([1,2,3,5.0], 1.2) == UnitRange{Int}[1:3, 4:4]
@test degenerate_ranges([0,2,3,4.0], 1.2) == UnitRange{Int}[1:1, 2:4]
@test degenerate_ranges([0,2,3,5.0], 1.2) == UnitRange{Int}[1:1, 2:3, 4:4]

function test_slater_determinants(lattice::AbstractLattice, N_up::Int, N_dn::Int, ϵ)
    apply_hamiltonian = hubbard_hamiltonian(t=1)
    hs = HubbardHilbertSpace(lattice)
    seed_state!(hs, N_up, N_dn)

    rst = RepresentativeStateTable(hs, apply_hamiltonian)

    # Determine the energy and momentum of each possible band filling.  Store
    # everything in the `band_fillings` dict, which is keyed by total momentum.
    @assert isbravais(lattice)
    band_fillings = Dict{Vector{Rational{Int}}, Vector{Tuple{Float64, Vector{Int}, Vector{Int}}}}()
    for filled_k_up_indices in combinations(eachmomentumindex(lattice), N_up)
        for filled_k_dn_indices in combinations(eachmomentumindex(lattice), N_dn)
            filled_orbitals = map(k_idx -> momentum(lattice, k_idx),
                                  [filled_k_up_indices; filled_k_dn_indices])
            energy = mapreduce(ϵ, +, filled_orbitals)
            total_momentum = rem.(reduce(+, filled_orbitals), 1)
            v = get!(band_fillings, total_momentum) do
                Tuple{Float64, Vector{Int}, Vector{Int}}[]
            end
            push!(v, (energy, filled_k_up_indices, filled_k_dn_indices))
        end
    end
    @assert length(band_fillings) == nmomenta(lattice)

    # Sort by energy at each momentum.  This will help us soon determine
    # nearly-adjacent energy levels.
    for v in values(band_fillings)
        sort!(v, by=(x -> x[1]))
    end
    @assert mapreduce(length, +, values(band_fillings)) == length(hs.indexer)

    total_momenta = [rem.(momentum(lattice, k_idx, N_up + N_dn), 1) for k_idx in eachmomentumindex(lattice)]

    root_V = sqrt(length(lattice))

    # Construct each Slater determinant wavefunction
    slater_wfs = [Array{Complex128}(length(hs.indexer), length(band_fillings[k])) for k in total_momenta]
    for (i, state) in enumerate(hs.indexer)
        # First figure out the positions of the particles
        r_up = find(x -> x & 1 != 0, state)
        r_dn = find(x -> x & 2 != 0, state)
        @assert length(r_up) == N_up
        @assert length(r_dn) == N_dn

        # Calculate each Slater determinant at these positions
        for (total_k, slater_wfs_k) in zip(total_momenta, slater_wfs)
            for (j, (energy, filled_k_up_indices, filled_k_dn_indices)) in enumerate(band_fillings[total_k])
                # Construct and calculate the two determinants with proper normalization.
                up_mat = Complex128[exp(im * kdotr(lattice, k, r)) / root_V for r in r_up, k in filled_k_up_indices]
                dn_mat = Complex128[exp(im * kdotr(lattice, k, r)) / root_V for r in r_dn, k in filled_k_dn_indices]
                slater_wfs_k[i, j] = det(up_mat) * det(dn_mat)
            end
        end
    end

    # Diagonalize each momentum sector, and check the energies and overlaps
    for k_idx in eachmomentumindex(hs.lattice)
        slater_energies = [energy for (energy, ku, kd) in band_fillings[total_momenta[k_idx]]]
        diagsect = DiagonalizationSector(rst, 1, k_idx)
        length(diagsect) != 0 || continue
        reduced_ham = full(construct_reduced_hamiltonian(diagsect))
        @test vecnorm(reduced_ham - reduced_ham') < 1e-5
        fact = eigfact(Hermitian((reduced_ham + reduced_ham') / 2))
        evals, evecs = fact[:values], fact[:vectors]

        # Check the energies
        @test evals ≈ slater_energies atol=1e-10

        # Check the overlaps
        i_check = 0
        for degenerate_range in degenerate_ranges(slater_energies, 1e-8)
            for i in degenerate_range
                i_check += 1
                @assert i == i_check

                ψ = get_full_psi(diagsect, @view evecs[:,i])
                # Project ψ into the subspace of Slater determinant eigenstates
                # at this energy.  It should remain unchanged by this
                # projection.
                ϕ = zeros(ψ)
                for j in degenerate_range
                    slater_wf = @view slater_wfs[k_idx][:,j]
                    ϕ .+= slater_wf .* dot(slater_wf, ψ)
                end
                @test ϕ ≈ ψ atol=1e-10
            end
        end
        @test i_check == length(evals)
    end
end

let
    hypercubic_ϵ(k) = -2 * mapreduce(kα -> cos(2π * kα), +, k)

    test_slater_determinants(ChainLattice([6]), 2, 0, hypercubic_ϵ)
    test_slater_determinants(ChainLattice([6]), 3, 3, hypercubic_ϵ)
    test_slater_determinants(ChainLattice([6], diagm([6]), [1//2]), 1, 0, hypercubic_ϵ)
    test_slater_determinants(ChainLattice([6], diagm([6]), [1//5]), 1, 0, hypercubic_ϵ)
    test_slater_determinants(ChainLattice([6], diagm([6]), [1//2]), 3, 3, hypercubic_ϵ)
    test_slater_determinants(ChainLattice([6], diagm([6]), [1//5]), 3, 3, hypercubic_ϵ)
    test_slater_determinants(ChainLattice([5]), 2, 3, hypercubic_ϵ)
    test_slater_determinants(SquareLattice([2, 3]), 3, 3, hypercubic_ϵ)

    #test_slater_determinants(TriangularLattice([2, 3]), 3, 3)
    #test_slater_determinants(TriangularLattice([2, 3], diagm([2,3]), [1//2, 1//3]), 3, 3)
end

function test_hubbard_abelian_spinflip(lattice, N_updn; t=1, U=2, kwargs...) # does not assume half filling
    hs = HubbardHilbertSpace(lattice)
    seed_state!(hs, N_updn, N_updn)
    apply_hamiltonian = hubbard_hamiltonian(t=t, U=U; kwargs...)
    rst = RepresentativeStateTable(hs, apply_hamiltonian, [spinflip_symmetry])

    full_ham = operator_matrix(hs, apply_hamiltonian)

    processed_length = 0
    for k_idx in eachmomentumindex(hs.lattice)
        for spinflip_idx in 0:1
            diagsect = DiagonalizationSector(rst, 1, k_idx, [spinflip_idx])
            length(diagsect) != 0 || continue
            processed_length += length(diagsect)
            reduced_ham = full(construct_reduced_hamiltonian(diagsect))
            @test vecnorm(reduced_ham - reduced_ham') < 1e-5
            fact = eigfact(Hermitian((reduced_ham + reduced_ham') / 2))
            evals, evecs = fact[:values], fact[:vectors]
            for i in 1:length(evals)
                eval = evals[i]
                evec = evecs[:,i]
                ψ = get_full_psi(diagsect, evec)

                @test eigenstate_badness(full_ham, eval, ψ) ≈ 0 atol=1e-8

                if i == 1
                    let L = length(lattice)
                        for L_A in 0:div(L, 2)
                            ent_cut1 = entanglement_entropy(Tracer(hs, 1:L_A), ψ)
                            ent_cut2 = entanglement_entropy(Tracer(hs, 1:L-L_A), ψ)
                            if ndimensions(hs.lattice) == 1
                                @test ent_cut1 ≈ ent_cut2 atol=1e-8
                            end
                        end
                    end
                end
            end
        end
    end
    @test processed_length == length(hs.indexer)
end

test_hubbard_abelian_spinflip(ChainLattice([6]), 2)
test_hubbard_abelian_spinflip(ChainLattice([6]), 3)
test_hubbard_abelian_spinflip(ChainLattice([2], diagm([2]), [1//3]), 1)
test_hubbard_abelian_spinflip(ChainLattice([4], diagm([4]), [1//3]), 2)
test_hubbard_abelian_spinflip(ChainLattice([6], diagm([6]), [1//2]), 3)
test_hubbard_abelian_spinflip(ChainLattice([6], diagm([6]), [1//5]), 3)
test_hubbard_abelian_spinflip(ChainLattice([6], diagm([6]), [1//5]), 3, t2=0.2)
test_hubbard_abelian_spinflip(ChainLattice([4], diagm([4]), [1//5]), 2, t2=0.2, J=0.3)

test_hubbard_abelian_spinflip(SquareLattice([2,3]), 3)
test_hubbard_abelian_spinflip(TriangularLattice([2,3]), 3)
# NOTE: The triangular lattice does not (yet) implement next-nearest
# neighbors, so this should fail.
@test_throws MethodError test_hubbard_abelian_spinflip(TriangularLattice([2,3]), 3, t2=0.2)

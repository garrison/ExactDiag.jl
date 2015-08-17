@test ExactDiag.permutation_parity(Int[]) == 0
@test ExactDiag.permutation_parity([1]) == 0
@test ExactDiag.permutation_parity([1,2]) == 0
@test ExactDiag.permutation_parity([2,1]) == 1
@test ExactDiag.permutation_parity([1,2,3]) == 0
@test ExactDiag.permutation_parity([3,1,2]) == 0
@test ExactDiag.permutation_parity([1,3,2]) == 1

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

# With abelian symmetries
let L = 4
    hs = HubbardHilbertSpace(ChainLattice([L]))
    seed_state!(hs, div(L, 2), div(L, 2))
    apply_hamiltonian = hubbard_hamiltonian(t=1, U=2)
    rst = RepresentativeStateTable(hs, apply_hamiltonian, [spinflip_symmetry])

    full_ham = operator_matrix(hs, apply_hamiltonian)

    processed_length = 0
    for k_idx in 1:nmomenta(hs.lattice)
        for spinflip_idx in 0:1
            diagsect = DiagonalizationSector(rst, 1, k_idx, [spinflip_idx])
            processed_length += length(diagsect)
            reduced_ham = full(construct_reduced_hamiltonian(diagsect))
            fact = eigfact(Hermitian((reduced_ham + reduced_ham') / 2))
            evals, evecs = fact[:values], fact[:vectors]
            for i in 1:length(evals)
                eval = evals[i]
                evec = evecs[:,i]
                ψ = get_full_psi(diagsect, evec)

                @test_approx_eq_eps eigenstate_badness(full_ham, eval, ψ) 0 1e-8
                @test_approx_eq_eps eigenstate_badness(hs, apply_hamiltonian, eval, ψ) 0 1e-8
                check_eigenstate(full_ham, eval, ψ)
                @test_throws InexactError check_eigenstate(full_ham, eval + 1, ψ)

                if i == 1
                    for L_A in 0:div(L, 2)
                        ent_cut1 = entanglement_entropy(Tracer(hs, 1:L_A), ψ)
                        ent_cut2 = entanglement_entropy(Tracer(hs, 1:L-L_A), ψ)
                        # FIXME: test against known results
                        @test_approx_eq_eps ent_cut1 ent_cut2 1e-8
                    end
                end
            end
            #println("$L\t$(k_idx-1)\t$(1-2*spinflip_idx)\t$(evals[1])")
        end
    end
    @test processed_length == length(hs.indexer)
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

# Slater determinants
let
    L = 6
    lattice = ChainLattice([6])
    apply_hamiltonian = hubbard_hamiltonian(t=1)
    hs = HubbardHilbertSpace(lattice)
    seed_state!(hs, div(L, 2), div(L, 2))

    rst = RepresentativeStateTable(hs, apply_hamiltonian, [spinflip_symmetry])
    full_ham = operator_matrix(hs, apply_hamiltonian)

    # Find the ground state wavefunction
    local gs_eval, gs_evec
    gs_eval = maxintfloat(Float64) # fixme
    for k_idx in 1:nmomenta(hs.lattice)
        for spinflip_idx in 0:1
            diagsect = DiagonalizationSector(rst, 1, k_idx, [spinflip_idx])
            length(diagsect) != 0 || continue
            reduced_ham = full(construct_reduced_hamiltonian(diagsect))
            fact = eigfact(Hermitian((reduced_ham + reduced_ham') / 2))
            evals, evecs = fact[:values], fact[:vectors]
            for i in [1]#1:length(evals)
                eval = evals[i]
                evec = evecs[:,i]
                ψ = get_full_psi(diagsect, evec)
                if eval < gs_eval
                    gs_eval = eval
                    gs_evec = ψ
                end
            end
        end
    end
    debug && @show gs_eval

    # Construct the Slater determinant wavefunction
    slater = Array(Complex128, length(hs.indexer))
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
        mat1 = Array(Complex128, pn, pn)
        mat2 = Array(Complex128, pn, pn)
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
    @test_approx_eq abs(overlap) 1
end

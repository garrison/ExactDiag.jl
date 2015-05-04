let
    indexer = IndexedArray{Vector{Int}}(Vector{Int}[[1],[0]])
    hs = SpinHalfHilbertSpace(ChainLattice([1]), indexer)
    @test operator_matrix(hs, apply_σx, 1) == [0 1; 1 0]
    @test operator_matrix(hs, apply_σy, 1) == [0 -im; im 0]
    @test operator_matrix(hs, apply_σz, 1) == [1 0; 0 -1]
end

let
    indexer = IndexedArray{Vector{Int}}(Vector{Int}[[1,1],[0,1],[1,0],[0,0]])
    hs = SpinHalfHilbertSpace(ChainLattice([2]), indexer)

    @test operator_matrix(hs, apply_σx, 1) * operator_matrix(hs, apply_σx, 2) + operator_matrix(hs, apply_σy, 1) * operator_matrix(hs, apply_σy, 2) == sparse([3,2],[2,3],[2,2], 4, 4)
    @test operator_matrix(hs, apply_σxσx_σyσy, 1, 2) == sparse([3,2],[2,3],[2,2], 4, 4)
    @test operator_matrix(hs, apply_σxσx_σyσy, 1, 1) == spdiagm([2,2,2,2])

    @test operator_matrix(hs, apply_σzσz, 1, 1) == spdiagm([1, 1, 1, 1])
    @test operator_matrix(hs, apply_σzσz, 1, 2) == spdiagm([1, -1, -1, 1])
    @test operator_matrix(hs, apply_σz, 1) * operator_matrix(hs, apply_σz, 2) == spdiagm([1, -1, -1, 1])
end

function test_disordered_hamiltonian(lattice, expected_gs, expected_Sz)
    h_z = [-0.9994218963834927, -0.49906680067568954, 0.3714572638372098, 0.9629810631305735, 0.19369581339829733, -0.7411831242535816, -0.061683656841222456, 0.30784629029574884, -0.42077926330644844, 0.25473615736727395, 0.12683294253359123, -0.6640580830314939]
    apply_hamiltonian = spin_half_hamiltonian(J1=1, h_z=h_z)
    indexer = IndexedArray{Vector{Int}}()
    hs = SpinHalfHilbertSpace(lattice, indexer)
    seed_state!(hs, div(length(lattice), 2))
    mat = operator_matrix(hs, apply_hamiltonian)
    @test ishermitian(mat)
    #rv = eigs(Hermitian(mat), nev=1, which=:SR)
    evals, evecs = eigs(mat, which=:SR)
    for (i, eval) in enumerate(evals)
        @test_approx_eq_eps (sum(mat * evecs[:,i] - eval * evecs[:,i])) 0 1e-10
    end
    #println(evals)
    #println(evecs)

    # Test GS energy
    @test_approx_eq evals[1] expected_gs

    # Test some correlators in the GS
    evec = evecs[:,1]
    for x1 in 1:length(lattice)
        vec = zeros(evec)
        for (j, state) in enumerate(hs.indexer)
            apply_Sz(hs, j, x1) do i, v
                vec[i] += v * evec[j]
            end
        end
        exv = dot(vec, evec)
        @test_approx_eq_eps exv expected_Sz[x1] 1e-10
    end
end

pbc_Sz = [0.251970534661742, -0.014905816107976, 0.0138790327761257, -0.274704496076216, 0.016042987433838, 0.067019907241627, 0.107468173924465, -0.231028029504729, 0.25946797753064, -0.236502949591289, 0.0575550494278513, -0.0162623717160799]
test_disordered_hamiltonian(ChainLattice([12]), -5.75814398110789, pbc_Sz)

obc_Sz = [0.380927022651224, -0.136056762511492, 0.118088125191709, -0.324427554030101, -0.0293082646011701, 0.080280872605819, 0.107382672186401, -0.196379958994102, 0.189553312888283, -0.203159954875788, -0.190071091118365, 0.203171580607582]
test_disordered_hamiltonian(ChainLattice([12], diagm([0])), -5.63552961749324, obc_Sz)

# FIXME: Also test SzSz and S+S- correlators (see https://github.com/simple-dmrg/sophisticated-dmrg/blob/master/test_disordered_heisenberg.py)

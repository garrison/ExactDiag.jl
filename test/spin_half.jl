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

# Test that spin operators have commutation relations as expected.
commutator(a,b) = a*b - b*a
anticommutator(a,b) = a*b + b*a
function test_pauli_commutation_relations(lattice)
    hs = SpinHalfHilbertSpace(lattice)
    seed_state!(hs, 0)

    # Populate all states
    operator_matrix(hs, spin_half_hamiltonian(h_x=1))
    @assert length(hs.indexer) == 2 ^ length(lattice)

    apply_sigma = [apply_σx, apply_σy, apply_σz]
    δ(i,j) = i == j ? 1 : 0
    for i in 1:length(lattice)
        σ_i = [operator_matrix(hs, apply_sigma[a], i) for a in 1:3]
        for j in 1:length(lattice)
            σ_j = [operator_matrix(hs, apply_sigma[a], j) for a in 1:3]
            for a in 1:3
                @test commutator(σ_i[a], σ_j[rem(a, 3) + 1]) == 2 * im * σ_j[rem(a + 1, 3) + 1] * δ(i,j)
            end
        end
        for a in 1:3
            for b in 1:3
                @test anticommutator(σ_i[a], σ_i[b]) == 2 * eye(length(hs.indexer)) * δ(a,b)
            end
        end
    end
end
test_pauli_commutation_relations(ChainLattice([1]))
test_pauli_commutation_relations(ChainLattice([4]))
test_pauli_commutation_relations(SquareLattice([2,2]))

function test_disordered_hamiltonian(lattice, expected_gs, expected_Sz, expected_SzSz, expected_SpSm)
    h_z = [-0.9994218963834927, -0.49906680067568954, 0.3714572638372098, 0.9629810631305735, 0.19369581339829733, -0.7411831242535816, -0.061683656841222456, 0.30784629029574884, -0.42077926330644844, 0.25473615736727395, 0.12683294253359123, -0.6640580830314939]
    apply_hamiltonian = spin_half_hamiltonian(J1=1, h_z=h_z)
    hs = SpinHalfHilbertSpace(lattice)
    seed_state!(hs, div(length(lattice), 2))
    mat = operator_matrix(hs, apply_hamiltonian)
    @test ishermitian(mat)
    #rv = eigs(Hermitian(mat), nev=1, which=:SR)
    evals, evecs = eigs(mat, which=:SR)
    for (i, eval) in enumerate(evals)
        @test sum(mat * evecs[:,i] - eval * evecs[:,i]) ≈ 0 atol=1e-10
    end

    # Test GS energy
    @test evals[1] ≈ expected_gs

    # Test some correlators in the GS
    evec = evecs[:,1]
    for x1 in 1:length(lattice)
        @test expectval(hs, evec, apply_Sz, x1) ≈ expected_Sz[x1] atol=1e-10
        for x2 in 1:length(lattice)
            @test expectval(hs, evec, apply_SzSz, x1, x2) ≈ expected_SzSz[x1, x2] atol=1e-10
            @test expectval(hs, evec, apply_SpSm, x1, x2) ≈ expected_SpSm[x1, x2] atol=1e-10
            x1 != x2 && @test expectval(hs, evec, apply_SmSp, x2, x1) ≈ expected_SpSm[x1, x2] atol=1e-10
        end
    end
end

pbc_energy = -5.75814398110789
pbc_Sz = [0.251970534661742, -0.014905816107976, 0.0138790327761257, -0.274704496076216, 0.016042987433838, 0.067019907241627, 0.107468173924465, -0.231028029504729, 0.25946797753064, -0.236502949591289, 0.0575550494278513, -0.0162623717160799]
pbc_SzSz = [0.250000000000002 -0.0948097047285464 0.0317505952964268 -0.0967567018203554 0.0162116835649695 0.00149409649794003 0.0423706517828481 -0.0733589121869654 0.0788434878374333 -0.0883253601880022 0.0340647567836856 -0.101484592839436
            -0.0948097047285464 0.250000000000002 -0.175111575226115 0.0282776445697974 -0.0111241876313048 0.00459950022381238 -0.0214893056584108 0.00691565687996494 -0.0124786001392513 0.0148990682539378 -0.00960632965949583 0.0199278331156101
            0.0317505952964268 -0.175111575226115 0.250000000000002 -0.089452159749666 0.0131785070943147 -0.013878194993777 0.0185159986221862 -0.0102806765400002 0.00959678515398784 -0.0164749639135959 -0.00100303270555681 -0.0168412830382062
            -0.0967567018203554 0.0282776445697974 -0.089452159749666 0.250000000000002 -0.0930788242020511 0.00971420235789359 -0.0646050129713476 0.0757397481064414 -0.0865628793117161 0.0779696730009117 -0.0301730264110209 0.0189273364311113
            0.0162116835649695 -0.0111241876313048 0.0131785070943147 -0.0930788242020511 0.250000000000002 -0.155883028146471 0.0289733038239265 -0.0432154413327269 0.0205541574692712 -0.0235689305646957 0.016885870822518 -0.0189331108977525
            0.00149409649794003 0.00459950022381238 -0.013878194993777 0.00971420235789359 -0.155883028146471 0.250000000000002 -0.112633836795471 0.021181867142089 -0.00709414272531618 0.00195153075048392 -0.0132474736007826 0.0137954792895957
            0.0423706517828481 -0.0214893056584108 0.0185159986221862 -0.0646050129713476 0.0289733038239265 -0.112633836795471 0.250000000000002 -0.144604381285307 0.0628195068612626 -0.0592733890613476 0.021527878295374 -0.0216014136137154
            -0.0733589121869654 0.00691565687996494 -0.0102806765400002 0.0757397481064414 -0.0432154413327269 0.021181867142089 -0.144604381285307 0.250000000000002 -0.154323757349207 0.0890830352801109 -0.0398906861503054 0.0227535474359047
            0.0788434878374333 -0.0124786001392513 0.00959678515398784 -0.0865628793117161 0.0205541574692712 -0.00709414272531618 0.0628195068612626 -0.154323757349207 0.250000000000002 -0.173880927079871 0.0499271888282659 -0.0374008195448612
            -0.0883253601880022 0.0148990682539378 -0.0164749639135959 0.0779696730009117 -0.0235689305646957 0.00195153075048392 -0.0592733890613476 0.0890830352801109 -0.173880927079871 0.250000000000002 -0.110860953171183 0.0384812166932486
            0.0340647567836856 -0.00960632965949583 -0.00100303270555681 -0.0301730264110209 0.016885870822518 -0.0132474736007826 0.021527878295374 -0.0398906861503054 0.0499271888282659 -0.110860953171183 0.250000000000002 -0.167624193031501
            -0.101484592839436 0.0199278331156101 -0.0168412830382062 0.0189273364311113 -0.0189331108977525 0.0137954792895957 -0.0216014136137154 0.0227535474359047 -0.0374008195448612 0.0384812166932486 -0.167624193031501 0.250000000000002]
pbc_SpSm = [0.751970534661746 -0.236707243119345 0.137547206113695 -0.1040797023312 0.0754002612356565 -0.0703476956346235 0.0705336897112939 -0.0689946405134317 0.0713172226772661 -0.108328176770239 0.147578066216467 -0.255219584725692
            -0.236707243119345 0.485094183892028 -0.377145944163698 0.135669064979232 -0.0990781261905124 0.0806447621714639 -0.0881586121175788 0.0682511202227959 -0.0623435422086656 0.0704928171089095 -0.0958036733792964 0.110660640080166
            0.137547206113695 -0.377145944163698 0.51387903277613 -0.232147139946612 0.115269516240226 -0.0906760299912435 0.0815474629836489 -0.0675013774015044 0.0607593825487087 -0.0754450829312008 0.0924020944038563 -0.102228823699145
            -0.1040797023312 0.135669064979232 -0.232147139946612 0.225295503923788 -0.240239221381729 0.123850250199894 -0.126328662058452 0.0827807090585064 -0.0673826019482099 0.0630104936793127 -0.0750093986615943 0.0711928141642749
            0.0754002612356565 -0.0990781261905124 0.115269516240226 -0.240239221381729 0.516042987433842 -0.348754526490564 0.173753140324495 -0.130377171992866 0.0773722643654037 -0.0770493270005411 0.0675903987731788 -0.0732122306357085
            -0.0703476956346235 0.0806447621714639 -0.0906760299912435 0.123850250199894 -0.348754526490564 0.567019907241631 -0.283207519112753 0.114471020257401 -0.080214753433728 0.065126437244307 -0.0656072861880371 0.0597662971832623
            0.0705336897112939 -0.0881586121175788 0.0815474629836489 -0.126328662058452 0.173753140324495 -0.283207519112753 0.607468173924469 -0.292093227075039 0.105875571659179 -0.102090452738707 0.0767630836362514 -0.078046483501324
            -0.0689946405134318 0.068251120222796 -0.0675013774015044 0.0827807090585064 -0.130377171992866 0.114471020257401 -0.292093227075039 0.268971970495275 -0.245172192343923 0.103683994325965 -0.0832897156677408 0.0689294038267282
            0.0713172226772661 -0.0623435422086656 0.0607593825487086 -0.0673826019482099 0.0773722643654037 -0.080214753433728 0.105875571659179 -0.245172192343923 0.759467977530644 -0.284715231600668 0.102943968333094 -0.10173422795094
            -0.108328176770239 0.0704928171089095 -0.0754450829312008 0.0630104936793127 -0.0770493270005412 0.0651264372443071 -0.102090452738707 0.103683994325965 -0.284715231600668 0.263497050408715 -0.244595812065143 0.130997075137417
            0.147578066216467 -0.0958036733792964 0.0924020944038563 -0.0750093986615943 0.0675903987731788 -0.0656072861880371 0.0767630836362514 -0.0832897156677407 0.102943968333094 -0.244595812065143 0.557555049427856 -0.364990495818253
            -0.255219584725692 0.110660640080166 -0.102228823699145 0.071192814164275 -0.0732122306357085 0.0597662971832623 -0.078046483501324 0.0689294038267282 -0.10173422795094 0.130997075137417 -0.364990495818253 0.483737628283924]
test_disordered_hamiltonian(ChainLattice([12]), pbc_energy, pbc_Sz, pbc_SzSz, pbc_SpSm)

obc_energy = -5.63552961749324
obc_Sz = [0.380927022651224, -0.136056762511492, 0.118088125191709, -0.324427554030101, -0.0293082646011701, 0.080280872605819, 0.107382672186401, -0.196379958994102, 0.189553312888283, -0.203159954875788, -0.190071091118365, 0.203171580607582]
obc_SzSz = [0.250000000000003 -0.126221187261807 0.0443942828827648 -0.144885160715301 -0.00277611796409998 0.019921122619248 0.0444460389387958 -0.0816634579877597 0.0754800493441368 -0.0823284490754211 -0.0713711415016831 0.0750040207211232
            -0.126221187261807 0.250000000000003 -0.166135440836618 0.0493678830282252 0.00481254874416006 -0.0149862613351064 -0.0168749629002306 0.0220704553195463 -0.0238677691507445 0.0243437577921975 0.0267673847910734 -0.0292764081906997
            0.0443942828827648 -0.166135440836618 0.250000000000003 -0.123696769975193 0.00947211939657975 -0.004387975617772 0.0200553848794982 -0.0279319539144804 0.02467555794653 -0.0270251675190035 -0.0220609403011598 0.0226409030588511
            -0.144885160715301 0.0493678830282252 -0.123696769975193 0.250000000000003 -0.0436421178972396 -0.00857574259218155 -0.0483305916506317 0.0685888352235739 -0.0644175946079963 0.0689344500167698 0.061239715263393 -0.0645829060934216
            -0.00277611796409998 0.00481254874416006 0.00947211939657975 -0.0436421178972396 0.250000000000003 -0.184816900536818 0.0320794358485881 -0.0487543774015821 0.016760800001719 -0.0254251381307306 0.0123521524453922 -0.0200624045059725
            0.019921122619248 -0.0149862613351064 -0.004387975617772 -0.00857574259218155 -0.184816900536818 0.250000000000003 -0.0886642798874285 0.0253300389687851 -0.00412142272581209 0.00476055009358912 -0.0198849302737381 0.025425801287231
            0.0444460389387958 -0.0168749629002306 0.0200553848794982 -0.0483305916506317 0.0320794358485881 -0.0886642798874285 0.250000000000003 -0.173029161705678 0.0609907922578787 -0.0714135684482618 -0.011775625817614 0.00251653848508151
            -0.0816634579877597 0.0220704553195463 -0.0279319539144804 0.0685888352235739 -0.0487543774015821 0.0253300389687851 -0.173029161705678 0.250000000000003 -0.120369637723445 0.0839802323956296 0.0257895063248239 -0.0240104794994157
            0.0754800493441368 -0.0238677691507445 0.02467555794653 -0.0644175946079963 0.016760800001719 -0.00412142272581209 0.0609907922578787 -0.120369637723445 0.250000000000003 -0.20447566943872 -0.0103066955844427 -0.000348410319107052
            -0.0823284490754211 0.0243437577921975 -0.0270251675190035 0.0689344500167698 -0.0254251381307306 0.00476055009358912 -0.0714135684482618 0.0839802323956296 -0.20447566943872 0.250000000000003 -0.0123968840442135 -0.00895411364183893
            -0.0713711415016831 0.0267673847910734 -0.0220609403011598 0.061239715263393 0.0123521524453922 -0.0198849302737381 -0.011775625817614 0.0257895063248239 -0.0103066955844427 -0.0123968840442135 0.250000000000003 -0.228352541301834
            0.0750040207211232 -0.0292764081906997 0.0226409030588511 -0.0645829060934216 -0.0200624045059725 0.025425801287231 0.00251653848508151 -0.0240104794994157 -0.000348410319107052 -0.00895411364183893 -0.228352541301834 0.250000000000003]
obc_SpSm = [0.88092702265123 -0.262661506108124 0.14116753056054 -0.106499804009335 0.0534885199582284 -0.0547254730183054 0.0389136440318317 -0.0378000406208441 0.0203259967167556 -0.0275377468671827 0.0154459653416028 -0.0157191000551504
            -0.262661506108124 0.363943237488514 -0.366347661131153 0.141599656140122 -0.0721219511685196 0.0685010854327646 -0.0544684997020737 0.0486077728360587 -0.0256826119248349 0.0333219425519274 -0.0178401278243851 0.0186595479705245
            0.14116753056054 -0.366347661131153 0.618088125191715 -0.259969594962114 0.0937046061087862 -0.0860700033246551 0.0555686574971302 -0.0525756084128191 0.0277749850981131 -0.0371363305270252 0.0207197497937901 -0.0209779658795406
            -0.106499804009335 0.141599656140122 -0.259969594962114 0.175572445969905 -0.169464482183663 0.111097295542631 -0.0849996732052932 0.068779985708525 -0.0351402135149769 0.0435247904047193 -0.0224463805679564 0.0237387702508952
            0.0534885199582284 -0.0721219511685196 0.0937046061087862 -0.169464482183663 0.470691735398836 -0.392796787620917 0.162724720948378 -0.146937171125042 0.0685925655317263 -0.0890076598318992 0.0461514107515209 -0.0477428675615197
            -0.0547254730183054 0.0685010854327646 -0.0860700033246551 0.111097295542631 -0.392796787620917 0.580280872605825 -0.23038885285509 0.113635799710859 -0.0572352443782531 0.0632405366018826 -0.0330514824586447 0.0334449788720193
            0.0389136440318317 -0.0544684997020737 0.0555686574971302 -0.0849996732052932 0.162724720948378 -0.23038885285509 0.607382672186407 -0.346290824830012 0.108283552025865 -0.13356906288566 0.0665814977597425 -0.066080213271577
            -0.0378000406208441 0.0486077728360587 -0.0525756084128191 0.068779985708525 -0.146937171125042 0.113635799710859 -0.346290824830012 0.303620041005904 -0.199174520212824 0.114552984120296 -0.0532381287522677 0.0517904540767088
            0.0203259967167556 -0.0256826119248349 0.0277749850981131 -0.0351402135149769 0.0685925655317263 -0.0572352443782531 0.108283552025865 -0.199174520212824 0.689553312888288 -0.373142162336678 0.10237370314646 -0.107709400425106
            -0.0275377468671827 0.0333219425519274 -0.0371363305270252 0.0435247904047193 -0.0890076598318992 0.0632405366018826 -0.13356906288566 0.114552984120296 -0.373142162336678 0.296840045124218 -0.137610588198593 0.110816804541974
            0.0154459653416028 -0.0178401278243851 0.0207197497937901 -0.0224463805679564 0.0461514107515209 -0.0330514824586447 0.0665814977597425 -0.0532381287522677 0.10237370314646 -0.137610588198593 0.309928908881641 -0.421728143640927
            -0.0157191000551504 0.0186595479705245 -0.0209779658795406 0.0237387702508952 -0.0477428675615197 0.0334449788720193 -0.066080213271577 0.0517904540767088 -0.107709400425106 0.110816804541974 -0.421728143640927 0.703171580607587]
test_disordered_hamiltonian(ChainLattice([12], diagm([0])), obc_energy, obc_Sz, obc_SzSz, obc_SpSm)

function my_1d_translate(state)
    state = copy(state)
    return unshift!(state, pop!(state))
end

function test_1d_translation_invariant_hamiltonian(lattice)
    apply_hamiltonian = spin_half_hamiltonian(J1=1)
    hs = SpinHalfHilbertSpace(lattice)
    seed_state!(hs, div(length(lattice), 2))
    mat = operator_matrix(hs, apply_hamiltonian)
    @test ishermitian(mat)
    zzz = HilbertSpaceTranslationCache(hs, 1)
    for j in 1:length(hs.indexer)
        i, η = translateη(zzz, j)
        @test η == 0
        @test hs.indexer[i] == my_1d_translate(hs.indexer[j])
        debug && println("$(hs.indexer[j])\t$(hs.indexer[i])\t$η")
    end
    # FIXME: compare with Bethe ansatz result and/or DMRG
    gs_energy, = eigs(mat, which=:SR, nev=1, ritzvec=false)
    @test gs_energy[1] ≈ -3.651093408937176
end
test_1d_translation_invariant_hamiltonian(ChainLattice([8]))

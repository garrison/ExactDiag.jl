typealias SpinHalfStateType Vector{Int}

immutable SpinHalfHilbertSpace{LatticeType<:AbstractSiteNetwork,IndexType<:AbstractIndexedArray{SpinHalfStateType}} <: HilbertSpace{SpinHalfStateType}
    lattice::LatticeType
    indexer::IndexType
end

SpinHalfHilbertSpace(lattice) = SpinHalfHilbertSpace(lattice, IndexedArray{SpinHalfStateType}())

statetype(::SpinHalfHilbertSpace) = SpinHalfStateType

get_σz(::SpinHalfHilbertSpace, site_state::Integer) = (site_state << 1) - 1
get_charge(::SpinHalfHilbertSpace, site_state::Integer) = 0

get_total_charge(::SpinHalfHilbertSpace, state) = 0 # because we cannot pick up any phase with twisted boundary conditions

function apply_σx(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    state = copy(hs.indexer[j])
    state[x1] $= 1
    i = findfirst!(hs.indexer, state)
    f(i, 1)
    nothing
end

function apply_σy(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    state = copy(hs.indexer[j])
    state[x1] $= 1
    i = findfirst!(hs.indexer, state)
    f(i, -im * get_σz(hs, state[x1]))
    nothing
end

function apply_σz(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    state = hs.indexer[j]
    f(j, get_σz(hs, state[x1]))
    nothing
end

function apply_σxσx(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    if x1 == x2
        f(j, 1)
    else
        state = copy(hs.indexer[j])
        state[x1] $= 1
        state[x2] $= 1
        i = findfirst!(hs.indexer, state)
        f(i, 1)
    end
    nothing
end

function apply_σzσz(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    state = hs.indexer[j]
    f(j, get_σz(hs, state[x1]) * get_σz(hs, state[x2]))
    nothing
end

function apply_σxσx_σyσy(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    if x1 == x2
        f(j, 2)
    else
        state = hs.indexer[j]
        if state[x1] $ state[x2] == 1
            state = copy(state)
            state[x1] $= 1
            state[x2] $= 1
            i = findfirst!(hs.indexer, state)
            f(i, 2)
        end
    end
    nothing
end

apply_σpσm_σmσp(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1) =
    apply_σxσx_σyσy(edapply(f, 1/2), hs, j, x1, x2)

function apply_σpσm(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    state = hs.indexer[j]
    if state[x2] == 1
        if x1 == x2
            f(j, 1)
        elseif state[x1] == 0
            state = copy(state)
            state[x1] $= 1
            state[x2] $= 1
            i = findfirst!(hs.indexer, state)
            f(i, 1)
        end
    end
    nothing
end

function apply_σmσp(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    state = hs.indexer[j]
    if state[x2] == 0
        if x1 == x2
            f(j, 1)
        elseif state[x1] == 1
            state = copy(state)
            state[x1] $= 1
            state[x2] $= 1
            i = findfirst!(hs.indexer, state)
            f(i, 1)
        end
    end
    nothing
end

apply_Sx(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer) =
    apply_σx(edapply(f, 1/2), hs, j, x1)

apply_Sy(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer) =
    apply_σy(edapply(f, 1/2), hs, j, x1)

apply_Sz(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer) =
    apply_σz(edapply(f, 1/2), hs, j, x1)

apply_SxSx(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1) =
    apply_σxσx(edapply(f, 1/4), hs, j, x1, x2)

apply_SzSz(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1) =
    apply_σzσz(edapply(f, 1/4), hs, j, x1, x2)

apply_SxSx_SySy(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1) =
    apply_σxσx_σyσy(edapply(f, 1/4), hs, j, x1, x2)

apply_SpSm(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1) =
    apply_σpσm(f, hs, j, x1, x2)

apply_SmSp(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1) =
    apply_σmσp(f, hs, j, x1, x2)

apply_SpSm_SmSp(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1) =
    apply_σpσm_σmσp(f, hs, j, x1, x2)

function apply_total_spin_operator(f, hs::SpinHalfHilbertSpace, j::Integer)
    state = hs.indexer[j]
    diagonal = 0.0

    for x in 1:length(hs.lattice)
        for x_r in 1:length(hs.lattice)
            # 0.5 * (S^+_i S^-_j + S^-_i S^+_j)
            if x == x_r
                diagonal += 0.5
            elseif state[x] $ state[x_r] == 1
                other = copy(state)
                other[x] $= 1
                other[x_r] $= 1
                s_i = findfirst!(hs.indexer, other)
                f(s_i, 0.5)
            end
            # S^z_i S^z_j
            diagonal += 0.25 * get_σz(hs, state[x]) * get_σz(hs, state[x_r])
        end
    end

    f(j, diagonal)
    nothing
end

function spin_half_hamiltonian(;
                               h_x::Union(Real,Vector)=0.0,
                               h_y::Union(Real,Vector)=0.0,
                               h_z::Union(Real,Vector)=0.0,
                               J1_xy::Real=0.0,
                               J1_z::Real=0.0,
                               J2_xy::Real=0.0,
                               J2_z::Real=0.0,
                               J1::Real=0.0,
                               J2::Real=0.0,
                               ϵ_total_spin::Real=0.0)
    if J1 != 0
        J1_xy == J1_z == 0 || throw(ArgumentError("If J1 is provided, J1_xy and J1_z must not be."))
        J1_xy = J1_z = J1
    end
    if J2 != 0
        J2_xy == J2_z == 0 || throw(ArgumentError("If J2 is provided, J2_xy and J2_z must not be."))
        J2_xy = J2_z = J2
    end

    len = 0
    for h_α in (h_x, h_y, h_z)
        if isa(h_α, Vector)
            len == 0 || len == length(h_α) || throw(ArgumentError("Disorder potentials must match in size."))
            len = length(h_α)
        end
    end

    return function apply_hamiltonian(f, hs::SpinHalfHilbertSpace, j::Integer)
        len == 0 || len == length(hs.lattice) || throw(ArgumentError("Lattice size does not match size of disorder potential."))

        # NOTE: When modifying, we sure to modify this outer conditional
        # as well!
        if h_x != 0 || h_y != 0 || h_z != 0
            for x1 in 1:length(hs.lattice)
                h_x != 0 && apply_Sx(edapply(f, getval(h_x, x1)), hs, j, x1)
                h_y != 0 && apply_Sy(edapply(f, getval(h_y, x1)), hs, j, x1)
                h_z != 0 && apply_Sz(edapply(f, getval(h_z, x1)), hs, j, x1)
            end
        end

        # NOTE: When modifying, we sure to modify this outer conditional
        # as well!
        if J1_xy != 0 || J1_z != 0
            neighbors(hs.lattice) do x1, x2, wrap
                J1_xy != 0 && apply_SxSx_SySy(edapply(f, J1_xy), hs, j, x1, x2)
                J1_z != 0 && apply_SzSz(edapply(f, J1_z), hs, j, x1, x2)
            end
        end

        # NOTE: When modifying, we sure to modify this outer conditional
        # as well!
        if J2_xy != 0 || J2_z != 0
            neighbors(hs.lattice, Val{2}) do x1, x2, wrap
                J2_xy != 0 && apply_SxSx_SySy(edapply(f, J2_xy), hs, j, x1, x2)
                J2_z != 0 && apply_SzSz(edapply(f, J2_z), hs, j, x1, x2)
            end
        end

        if ϵ_total_spin != 0
            apply_total_spin_operator(edapply(f, ϵ_total_spin), hs, j)
        end

        nothing
    end
end

function seed_state!(hs::SpinHalfHilbertSpace, N_up::Integer)
    if !(0 <= N_up <= length(hs.lattice))
        throw(ArgumentError("Invalid N_up provided for size $(length(hs.lattice)) lattice: $(N_up)"))
    end
    state = zeros(Int, length(hs.lattice))
    for i in 1:N_up
        state[i] = 1
    end
    findfirst!(hs.indexer, state)
    return hs
end

# seed_state, conserves_sz

function translateη(hs::SpinHalfHilbertSpace, ltrc::LatticeTranslationCache, j::Integer)
    @assert hs.lattice === ltrc.lattice
    oldstate = hs.indexer[j]
    state = zeros(oldstate)
    for (i, site_state) in enumerate(oldstate)
        j, η = translateη(ltrc, i)
        state[j] = site_state
    end
    return findfirst!(hs.indexer, state), 0//1
end

function spinflipη(hs::SpinHalfHilbertSpace, j::Integer)
    state = hs.indexer[j]
    i = findfirst!(hs.indexer, [x $ 1 for x in state])
    return i, 0//1
end

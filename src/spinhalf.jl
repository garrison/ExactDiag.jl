typealias SpinHalfStateType Vector{Int}

immutable SpinHalfHilbertSpace{LatticeType<:AbstractSiteNetwork,IndexType<:AbstractIndexedArray{SpinHalfStateType}} <: HilbertSpace
    lattice::LatticeType
    indexer::IndexType
end

statetype(::Type{SpinHalfHilbertSpace}) = SpinHalfStateType
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
apply_σxσx_σyσy() = nothing

function apply_Sx(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    apply_σx(hs, j, x1) do i, v
        f(i, v / 2)
    end
    nothing
end

function apply_Sy(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    apply_σy(hs, j, x1) do i, v
        f(i, v / 2)
    end
    nothing
end

function apply_Sz(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    apply_σz(hs, j, x1) do i, v
        f(i, v / 2)
    end
    nothing
end

function apply_SxSx(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    apply_σxσx(hs, j, x1, x2) do i, v
        f(i, v / 4)
    end
    nothing
end

function apply_SzSz(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    apply_σzσz(hs, j, x1, x2) do i, v
        f(i, v / 4)
    end
    nothing
end

function apply_SxSx_SySy(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    apply_σxσx_σyσy(hs, j, x1, x2) do i, v
        f(i, v / 4)
    end
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
                               J2::Real=0.0)
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

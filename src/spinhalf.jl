immutable SpinHalfHilbertSpace <: HilbertSpace
    lattice::AbstractLattice  # or we could get rid of this; or just say how many sites
    indexer
end

# FIXME
function apply_translation()
end

get_σz(site_state::Integer) = (site_state << 1) - 1

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
    f(i, 1 - (state[x1] << 1))
    nothing
end

function apply_σz(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    state = hs.indexer[j]
    f(j, get_σz(state[x1]))
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
    f(j, get_σz(state[x1]) * get_σz(state[x2]))
    nothing
end

# FIXME: check this!
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

immutable SpinHalfHamiltonian
    h_x::Float64
    h_y::Float64
    h_z::Float64
    J1_xy::Float64
    J1_z::Float64

    SpinHalfHamiltonian(; h_x::Real=0.0, h_y::Real=0.0, h_z::Real=0.0, J1_xy::Real=0.0, J1_z::Real=0.0) = new(h_x, h_y, h_z, J1_xy, J1_z)
end

function apply_hamiltonian(f, hs::SpinHalfHilbertSpace, j::Integer, ham::SpinHalfHamiltonian)
    # NOTE: When modifying, we sure to modify this outer conditional
    # as well!
    if ham.h_x != 0 || ham.h_y != 0 || ham.h_z != 0
        for x1 in 1:length(hs.lattice)
            if ham.h_x != 0
                apply_Sx(hs, j, x1) do i, v
                    f(i, ham.h_x * v)
                end
            end
            if ham.h_y != 0
                apply_Sx(hs, j, x1) do i, v
                    f(i, ham.h_y * v)
                end
            end
            if ham.h_z != 0
                apply_Sx(hs, j, x1) do i, v
                    f(i, ham.h_z * v)
                end
            end
        end
    end

    # NOTE: When modifying, we sure to modify this outer conditional
    # as well!
    if ham.J1_xy != 0 || ham.J1_z != 0
        neighborsη(hs.lattice) do x1, x2, η
            if ham.J1_xy != 0
                apply_SxSx_SySy(hs, j, x1, x2) do i, v
                    f(i, ham.J1_xy * v)
                end
            end
            if ham.J1_z != 0
                apply_SzSz(hs, j, x1, x2) do i, v
                    f(i, ham.J1_z * v)
                end
            end
        end
    end

    nothing
end

get_total_charge(hs::SpinHalfHilbertSpace, state) = 0

function seed_state!(hs::SpinHalfHilbertSpace, N_up::Integer)
    if !(0 <= N_up <= length(hs.lattice))
        throw(ArgumentError("Invalid N_up provided for size $(length(hs.lattice)) lattice: $(N_up)"))
    end
    state = zeros(Int, length(hs.lattice))
    for i in 1:N_up
        state[i] = 1
    end
    findfirst!(hs.indexer, state)
    return state
end

# seed_state, conserves_sz

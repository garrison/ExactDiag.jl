const BosonStateType = Vector{Int}

struct BosonHilbertSpace{LatticeType<:AbstractSiteNetwork,IndexType<:AbstractUniqueVector{BosonStateType}} <: HilbertSpace{BosonStateType}
    lattice::LatticeType
    indexer::IndexType
end

BosonHilbertSpace(lattice) =
    BosonHilbertSpace(lattice, UniqueVector{BosonStateType}())

statetype(::BosonHilbertSpace) = BosonStateType

get_charge(::BosonHilbertSpace, site_state::Integer) = site_state
get_total_charge(::BosonHilbertSpace, state::BosonStateType) =
    sum(state)
get_total_charge(hs::BosonHilbertSpace, stateidx::Int) = get_total_charge(hs, hs.indexer[stateidx])

struct BosonParameters
    J::Float64
    U::Float64
    θ::Float64

    function BosonParameters(;
                             J::Real=0.0,
                             U::Real=0.0,
                             θ::Real=0.0)
        new(J, U, θ)
    end
end

boson_hamiltonian(; kwargs...) =
    boson_hamiltonian(BosonParameters(; kwargs...))

function boson_hamiltonian(p::BosonParameters)
    return function apply_hamiltonian(f, hs::BosonHilbertSpace, s_j::Integer)
        state = hs.indexer[s_j]

        # Hopping
        neighborsη(hs.lattice) do x::Int, x_r::Int, η::Rational{Int}
            # FIXME: what to do with η ??
            # FIXME: Also, we are assuming 1D.
            # XXX FIXME: type stability; currently this always returns a Complex number!

            # Rightward hop
            if state[x] > 0
                other = copy(state)
                phase = exp(im * p.θ * other[x_r])
                factor = √other[x]
                other[x] -= 1
                other[x_r] += 1
                factor *= √other[x_r]
                s_i = findfirst!(isequal(other), hs.indexer)
                f(s_i, -p.J * factor * phase)
            end
            # Leftward hop
            if state[x_r] > 0
                other = copy(state)
                factor = √other[x_r]
                other[x_r] -= 1
                other[x] += 1
                factor *= √other[x]
                phase = exp(-im * p.θ * other[x_r])
                s_i = findfirst!(isequal(other), hs.indexer)
                f(s_i, -p.J * factor * phase)
            end
        end

        # Bose-Hubbard U
        if !iszero(p.U)
            cnt_U = 0
            for x in 1:length(hs.lattice)
                cnt_U += state[x] * (state[x] - 1)
            end
            f(s_j, 0.5 * p.U * cnt_U)
        end

        nothing
    end
end

function apply_n(f, hs::BosonHilbertSpace, j::Integer, x1::Integer)
    state = hs.indexer[j]
    f(j, state[x1])
    nothing
end
export apply_n

function seed_state!(hs::BosonHilbertSpace, N::Integer)
    state = zeros(Int, length(hs.lattice))
    state[1] = N # XXX FIXME
    findfirst!(isequal(state), hs.indexer)
    hs
end

function reflectionη(hs::BosonHilbertSpace, j::Integer)
    state = hs.indexer[j]
    i = findfirst!(isequal(reverse(state)), hs.indexer)
    return i, 0//1
end

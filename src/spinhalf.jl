const SpinHalfStateType{L} = SVector{L,Int}

struct SpinHalfHilbertSpace{L,LatticeType<:AbstractSiteNetwork,IndexType<:AbstractUniqueVector{SpinHalfStateType{L}}} <: HilbertSpace{SpinHalfStateType{L}}
    lattice::LatticeType
    indexer::IndexType

    function SpinHalfHilbertSpace(lattice::LatticeType, indexer::IndexType) where {L,LatticeType<:AbstractSiteNetwork,IndexType<:AbstractUniqueVector{SpinHalfStateType{L}}}
        length(lattice) == L || throw(ArgumentError("Size of indexer state must match size of lattice"))
        new{L,LatticeType,IndexType}(lattice, indexer)
    end
end

# XXX NOTE: not type stable since length(lattice) is not known at compile time
# (perhaps add an intermediate function that is type stable??)
SpinHalfHilbertSpace(lattice::AbstractSiteNetwork) = SpinHalfHilbertSpace(lattice, UniqueVector{SpinHalfStateType{length(lattice)}}())

statetype(::SpinHalfHilbertSpace{L}) where {L} = SpinHalfStateType{L}

get_σz(::SpinHalfHilbertSpace, site_state::Integer) = 1 - (site_state << 1)
get_charge(::SpinHalfHilbertSpace, site_state::Integer) = 0

get_total_charge(::SpinHalfHilbertSpace, state) = 0 # because we cannot pick up any phase with twisted boundary conditions

myflipbit(v::SVector, x::Integer) = setindex(v, xor(v[x], 1), x)
myflipbits(v::SVector, x1::Integer, x2::Integer) = myflipbit(myflipbit(v, x1), x2)

function apply_σ(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, σ::AbstractMatrix{<:Number})
    size(σ) == (2, 2) || throw(ArgumentError("σ matrix must be 2×2"))
    state = hs.indexer[j]
    other = myflipbit(state, x1)
    i = findfirst!(isequal(other), hs.indexer)
    f(j, σ[1+state[x1],1+state[x1]])
    f(i, σ[1+other[x1],1+state[x1]])
    nothing
end

function apply_σx(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    state = myflipbit(hs.indexer[j], x1)
    i = findfirst!(isequal(state), hs.indexer)
    f(i, 1)
    nothing
end

function apply_σy(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer)
    state = myflipbit(hs.indexer[j], x1)
    i = findfirst!(isequal(state), hs.indexer)
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
        state = myflipbits(hs.indexer[j], x1, x2)
        i = findfirst!(isequal(state), hs.indexer)
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
        if state[x1] ⊻ state[x2] == 1
            other = myflipbits(state, x1, x2)
            i = findfirst!(isequal(other), hs.indexer)
            f(i, 2)
        end
    end
    nothing
end

apply_σpσm_σmσp(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1) =
    apply_σxσx_σyσy(edapply(f, 1/2), hs, j, x1, x2)

function apply_σpσm(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    state = hs.indexer[j]
    if state[x2] == 0
        if x1 == x2
            f(j, 1)
        elseif state[x1] == 1
            other = myflipbits(state, x1, x2)
            i = findfirst!(isequal(other), hs.indexer)
            f(i, 1)
        end
    end
    nothing
end

function apply_σmσp(f, hs::SpinHalfHilbertSpace, j::Integer, x1::Integer, x2::Integer, η::Rational{Int}=0//1)
    state = hs.indexer[j]
    if state[x2] == 1
        if x1 == x2
            f(j, 1)
        elseif state[x1] == 0
            other = myflipbits(state, x1, x2)
            i = findfirst!(isequal(other), hs.indexer)
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
            elseif state[x] ⊻ state[x_r] == 1
                other = myflipbits(state, x, x_r)
                s_i = findfirst!(isequal(other), hs.indexer)
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
                               h_x::Union{Real,Vector}=0.0,
                               h_y::Union{Real,Vector}=0.0,
                               h_z::Union{Real,Vector}=0.0,
                               J1_xy::Real=0.0,
                               J1_z::Real=0.0,
                               J1_x::Real=0.0,
                               J2_xy::Real=0.0,
                               J2_z::Real=0.0,
                               J1::Real=0.0,
                               J2::Real=0.0,
                               ϵ_total_spin::Real=0.0)
    if J1 != 0
        J1_xy == J1_z == J1_x == 0 || throw(ArgumentError("If J1 is provided, J1_xy and J1_z must not be."))
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

    return function apply_hamiltonian(f, hs::SpinHalfHilbertSpace, j::Integer, sitefilter=(siteidx -> true))
        len == 0 || len == length(hs.lattice) || throw(ArgumentError("Lattice size does not match size of disorder potential."))

        # NOTE: When modifying, we sure to modify this outer conditional
        # as well!
        if h_x != 0 || h_y != 0 || h_z != 0
            for x1 in 1:length(hs.lattice)
                if sitefilter(x1)
                    h_x != 0 && apply_Sx(edapply(f, getval(h_x, x1)), hs, j, x1)
                    h_y != 0 && apply_Sy(edapply(f, getval(h_y, x1)), hs, j, x1)
                    h_z != 0 && apply_Sz(edapply(f, getval(h_z, x1)), hs, j, x1)
                end
            end
        end

        # NOTE: When modifying, we sure to modify this outer conditional
        # as well!
        if J1_xy != 0 || J1_z != 0 || J1_x != 0
            neighbors(hs.lattice) do x1, x2, wrap
                if sitefilter(x1) && sitefilter(x2)
                    J1_xy != 0 && apply_SxSx_SySy(edapply(f, J1_xy), hs, j, x1, x2)
                    J1_z != 0 && apply_SzSz(edapply(f, J1_z), hs, j, x1, x2)
                    J1_x != 0 && apply_SxSx(edapply(f, J1_x), hs, j, x1, x2)
                end
            end
        end

        # NOTE: When modifying, we sure to modify this outer conditional
        # as well!
        if J2_xy != 0 || J2_z != 0
            neighbors(hs.lattice, Val{2}) do x1, x2, wrap
                if sitefilter(x1) && sitefilter(x2)
                    J2_xy != 0 && apply_SxSx_SySy(edapply(f, J2_xy), hs, j, x1, x2)
                    J2_z != 0 && apply_SzSz(edapply(f, J2_z), hs, j, x1, x2)
                end
            end
        end

        if ϵ_total_spin != 0
            @assert all(sitefilter, 1:length(hs.lattice)) # XXX
            apply_total_spin_operator(edapply(f, ϵ_total_spin), hs, j)
        end

        nothing
    end
end

function seed_state!(hs::SpinHalfHilbertSpace{L}; N_up::Int) where {L}
    if !(0 <= N_up <= length(hs.lattice))
        throw(ArgumentError("Invalid N_up provided for size $(length(hs.lattice)) lattice: $(N_up)"))
    end
    state = ones(MVector{L,Int})
    for i in 1:N_up
        state[i] = 0
    end
    findfirst!(isequal(state), hs.indexer)
    return hs
end

# Deprecated 2018-09-15
@deprecate seed_state!(hs::SpinHalfHilbertSpace, N_up::Integer) seed_state!(hs, N_up=N_up)

# conserves_sz

function translateη(hs::SpinHalfHilbertSpace{L}, ltrc::LatticeTranslationCache, j::Integer) where {L}
    @assert hs.lattice === ltrc.lattice
    oldstate = hs.indexer[j]
    state = zeros(MVector{L,Int})
    for (i, site_state) in enumerate(oldstate)
        j, η = translateη(ltrc, i)
        state[j] = site_state
    end
    return findfirst!(isequal(state), hs.indexer), 0//1
end

function spinflipη(hs::SpinHalfHilbertSpace, j::Integer)
    state = hs.indexer[j]
    i = findfirst!(isequal(map(x -> x ⊻ 1, state)), hs.indexer)
    return i, 0//1
end

function reflectionη(hs::SpinHalfHilbertSpace, j::Integer)
    state = hs.indexer[j]
    i = findfirst!(isequal(reverse(state)), hs.indexer)
    return i, 0//1
end

struct SpinHalfFullIndexer{L} <: AbstractUniqueVector{SVector{L,Int}}
end

SpinHalfFullIndexer(L::Integer) = SpinHalfFullIndexer{L}()

Base.length(indexer::SpinHalfFullIndexer{L}) where {L} = 2 ^ L
Base.size(indexer::SpinHalfFullIndexer{L}) where {L} = (2 ^ L,)

function Base.getindex(indexer::SpinHalfFullIndexer{L}, i::Integer) where {L}
    checkbounds(indexer, i)
    i -= 1
    # FIXME: would be nice if we could use @SVector macro!!
    SVector{L,Int}([Int((i & (1 << (L - j))) != 0) for j in 1:L])
end

EqualTo = Base.Fix2{typeof(isequal)}

function Base.findfirst(p::EqualTo{<:StaticVector{L,Int}}, indexer::SpinHalfFullIndexer{L}) where {L}
    s = 1
    for i in 0:L-1
        v = p.x[L - i]
        @assert (v | 1) == 1
        s += v << i
    end
    s
end

UniqueVectors.findfirst!(p::EqualTo, indexer::SpinHalfFullIndexer) = findfirst(p, indexer)

using StaticArrays
using UniqueVectors
using Bravais
using ExactDiag

using JLD
using Combinatorics

using Compat
using Compat.Test

@static if VERSION >= v"0.7-"
    using SparseArrays
    using LinearAlgebra
    using Arpack
else
    eigen = eigfact
    ℯ = e # FIXME: when support for 0.6 is dropped, use MathConstants.e instead in hubbard.jl
end

debug = false

@inferred ExactDiag.exp_2πiη(0//1)
@test ExactDiag.exp_2πiη(0//1) == 1
@test ExactDiag.exp_2πiη(1//2) == -1
@test ExactDiag.exp_2πiη(-1//2) == -1
@test ExactDiag.exp_2πiη(-1//1) == 1

@test ExactDiag.exp_2πiη(-1//1) ≈ exp(-2π * im)
@test ExactDiag.exp_2πiη(1//2) ≈ exp(π * im)
@test ExactDiag.exp_2πiη(1//4) ≈ exp(π/2 * im)
@test ExactDiag.exp_2πiη(1//3) ≈ exp(2π/3 * im)
@test ExactDiag.exp_2πiη(1//3) ≈ exp(2π/3 * im)

@testset "Spin half         " begin include("spin_half.jl") end
@testset "Hubbard           " begin include("hubbard.jl") end
@testset "Abelian           " begin include("abelian.jl") end
@testset "Abelian spin-flip " begin include("abelian-spinflip.jl") end
@testset "Abelian reflection" begin include("abelian-reflection.jl") end
@testset "Time evolution    " begin include("time-evolution.jl") end

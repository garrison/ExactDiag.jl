using IndexedArrays
using Bravais
using ExactDiag
using Compat
using JLD
using Base.Test

debug = false

@inferred ExactDiag.exp_2πiη(0//1)
@test ExactDiag.exp_2πiη(0//1) == 1
@test ExactDiag.exp_2πiη(1//2) == -1
@test ExactDiag.exp_2πiη(-1//2) == -1
@test ExactDiag.exp_2πiη(-1//1) == 1
@test_approx_eq ExactDiag.exp_2πiη(-1//1) exp(-2π * im)
@test_approx_eq ExactDiag.exp_2πiη(1//2) exp(π * im)
@test_approx_eq ExactDiag.exp_2πiη(1//4) exp(π/2 * im)
@test_approx_eq ExactDiag.exp_2πiη(1//3) exp(2π/3 * im)
@test_approx_eq ExactDiag.exp_2πiη(1//3) exp(2π/3 * im)

include("spin_half.jl")
include("hubbard.jl")
include("abelian.jl")
include("abelian-spinflip.jl")
include("time-evolution.jl")

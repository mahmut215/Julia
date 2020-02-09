

using Distributions
using DelimitedFiles
using LinearAlgebra
using Knet
using Dates
using MAT

# Juno.@enter gabor

 # include("C:\\Users\\mah\\Desktop\\Julia\\mlpOrginal.jl")



include("C:\\Users\\mah\\Desktop\\mnist-mlp\\mlp.jl")

# include("C:\\Users\\mah\\Desktop\\mnist-mlp\\MlpKnet.jl")
clearconsole()
model =MLP.main("--hidden 2 2 2 2 --epochs 200000 --rapor 100000")

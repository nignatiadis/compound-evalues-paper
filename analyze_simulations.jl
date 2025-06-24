using Pkg
Pkg.activate(".")

using FileIO
using CategoricalArrays
using DataFrames
#using Plots
#using StatsPlots
using StatsBase
using LaTeXStrings
using CSV

dir = @__DIR__
data_dir = joinpath(dir, "simulation_results")

files = filter(x -> occursin(r".jld2$", x), readdir(data_dir))

method_res = DataFrame()

for file in files
    filepath = joinpath(data_dir, file)
    df = load(filepath, "method_res")  # Adjust "method_res" if the variable name differs
    method_res = vcat(method_res, df)
end

CSV.write("simulation_results.csv", method_res)


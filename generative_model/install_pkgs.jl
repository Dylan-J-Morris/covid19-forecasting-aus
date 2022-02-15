"""
Simple script to grab and install these packages into the global Julia install (easiest 
way of handling this).
"""

using Pkg 

Pkg.add("Distributions")
Pkg.add("Random" )
Pkg.add("Statistics" )
Pkg.add("LinearAlgebra")
Pkg.add("DataStructures")

Pkg.add("FileIO")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("Measures")
Pkg.add("Pyplot")
Pkg.add("Pipe")
Pkg.add("PDFmerger") # *NOTE* that this will not be added on phoenix 

Pkg.add("DelimitedFiles")
Pkg.add("Dates")

Pkg.add("Distributed")
Pkg.add("SharedArrays")
Pkg.add("ProgressMeter")
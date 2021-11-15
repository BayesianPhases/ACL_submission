# first go at modelling some data

using DynamicHMC, Turing
using Serialization
using Random
using MCMCChains
using LinearAlgebra
using ArgParse

#Random.seed!(0)

include("general.jl")
include("wrapped_cauchy.jl")
include("bundt.jl")
include("parse_command_line.jl")

@model function fitWrapped(angles,conditions,participants,electrodes,conditionN,participantN,electrodeN,::Type{T}=Float64) where {T}

    itpcC ~    filldist(Normal(),conditionN)
    itpcP ~    filldist(Normal(),participantN)
    itpcE ~    filldist(Normal(),electrodeN)
    itpc  ~ Normal()
    scale ~ Exponential(1.0)

    x = Array{Vector{T}}(undef,(participantN,electrodeN))

    for i in 1:participantN
    	for j in 1:electrodeN
            x[i,j] ~ Bundt()
	end
   end

   mu=0.0

    for i in 1:length(angles)

        thisCond=conditions[i]
        thisPart=participants[i]
	thisElec=electrodes[i]

        mu=atan(x[thisPart,thisElec][1],x[thisPart,thisElec][2])

	gamma = -log(logistic(-3+scale*(itpc+itpcC[thisCond]+itpcE[thisElec]+itpcP[thisPart])))

        angles[i] ~ WrappedCauchy(mu,gamma)

    end

end



parsedArgs=parseCommandLine()

runC=parsedArgs["runC"]
freqC=parsedArgs["freqC"]

numberP=16
electrodeN=32
conditionN=6

experiment=load(collect(5:4+numberP),freqC)

experiment=experiment[(experiment.freqC.==freqC) .& (experiment.electrode.<=electrodeN),:]

angles=experiment.angle

participants = [x-4 for x in experiment.participant]
conditions   = experiment.conditionC
electrodes   = experiment.electrode

iterations = parsedArgs["iterations"]
acceptance = 0.9

chain = sample(fitWrapped(angles,conditions,participants,electrodes,conditionN,numberP,electrodeN) , NUTS(acceptance) , MCMCThreads(),iterations,8)

chainName=parsedArgs["name"]

if parsedArgs["runC"]>0
    chainName=chainName*"_r"*string(runC)
end

chainName=chainName*"_f"*string(freqC)*"_m1.jls"

serialize(chainName, chain)    



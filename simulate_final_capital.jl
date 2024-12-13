using MDPs
import Base
using Revise
using RiskMDPs
using RiskMeasures
using Distributions
using Accessors
using DataFrames: DataFrame
using DataFramesMeta
using LinearAlgebra
using CSV: File
using Infiltrator
using CSV

"""
save the distribution of final capital to "data_bar.csv"; 
"plots.jl " contains the code of plotting the distribution of final capital
"""



"""
load a transient mdp from a csv file, 1-based index
"""
function load_mdp(input)
    mdp = DataFrame(input)
    mdp = @orderby(mdp, :idstatefrom, :idaction, :idstateto)
    
    statecount = max(maximum(mdp.idstatefrom), maximum(mdp.idstateto))
    states = Vector{IntState}(undef, statecount)
    state_init = BitVector(false for s in 1:statecount)

    for sd ∈ groupby(mdp, :idstatefrom)
        idstate = first(sd.idstatefrom)
        actions = Vector{IntAction}(undef, maximum(sd.idaction))
       
        action_init = BitVector(false for a in 1:length(actions))
        for ad ∈ groupby(sd, :idaction)
            idaction = first(ad.idaction)
            try 
            actions[idaction] = IntAction(ad.idstateto, ad.probability, ad.reward)
            catch e
                error("Error in state $(idstate-1), action $(idaction-1): $e")
            end
            action_init[idaction] = true
        end
        # report an error when there are missing indices
        all(action_init) ||
            throw(FormatError("Actions in state " * string(idstate - 1) *
                " that were uninitialized " * string(findall(.!action_init) .- 1 ) ))

        states[idstate] = IntState(actions)
        state_init[idstate] = true
    end
    IntMDP(states)
end


####

# evaluates the policy by simulation
function evaluate_policy(model::TabMDP, π::Vector{Int})

    # evaluation helper variables
    episodes = 7000
    horizon::Integer = 100
    # reward weights
    rweights::Vector{Float64} = 1.0 .^ (0:horizon-1)    
    
    # for the uniform initial state distribution
    states_number = state_count(model)
    returns = []
    # inistate = 1, no money; 
    for inistate in 2: (states_number -1)
        H = simulate(model, π, inistate, horizon, episodes)
        #@infiltrate

        # rets is a vector of the total rewards, size of rets = number of episodes
        rets = rweights' * H.rewards |> vec
        
        # returns is an array of returns for all episodes
        for r in rets
            push!(returns,r) 
        end
    end  
    returns
end


# save the distribution of final capitals to data_bar.csv
function save_bar_data(returns1,returns2,returns3,
    returns4)

    cmax = 8
    xmax = 8+2

 capital =[]
 for value in -1:cmax
   push!(capital,value)
 end

 count_1 = zeros(Int64,xmax)
 returns1 = map(Int,returns1)
 for i in returns1
    count_1[i+2] += 1
 end

 size1 = length(returns1)
 count_11 = zeros(Float64,xmax)
 for i in 1:xmax
    count_11[i] =  count_1[i] / size1 
 end


 returns2 = map(Int,returns2)
 count_2 = zeros(Int64,xmax)
 for i in returns2
    count_2[i+2] += 1
 end

 returns3 = map(Int,returns3)
 count_3 = zeros(Int64,xmax)
 for i in returns3
    count_3[i+2] += 1
 end

 returns4 = map(Int,returns4)
 count_4 = zeros(Int64,xmax)
 for i in returns4
    count_4[i+2] += 1
 end

 size2 = length(returns2)
 count_22 = zeros(Float64,xmax)
 for i in 1:xmax
    count_22[i] =  count_2[i] / size2
 end

 size3 = length(returns3)
 count_33 = zeros(Float64,xmax)
 for i in 1:xmax
    count_33[i] =  count_3[i] / size3
 end

 size4 = length(returns4)
 count_44 = zeros(Float64,xmax)
 for i in 1:xmax
    count_44[i] =  count_4[i] / size4 
 end

 filepath = joinpath(pwd(), "data_bar.csv");
 data = DataFrame(capital= capital, one = count_11, two = count_22, three = count_33,
               four = count_44)
 CSV.write(filepath, data)

end


function main()

    # The values below are for "7 mgp0.68.csv"
    # optimal policies for different risk level α of EVaR 

    #risk level of EVaR: α1 = 0.9
    π1 = [1, 2, 2, 2, 4, 3, 2, 1, 1]
    
    #risk level of EVaR:α2 = 0.7
    π2 = [1, 2, 2, 2, 2, 2, 2, 1, 1]
    
    #risk level of EVaR:α3 = 0.40
    π3 = [1, 3, 2, 2, 2, 2, 2, 1, 1]
    
    #risk level of EVaR:α4 = 0.20
    π4 = [1, 3, 4, 5, 6, 7, 8, 1, 1]


    filepath = joinpath(dirname(pwd()), "7 mgp0.68.csv")
    model = load_mdp(File(filepath))

    # compute returns for four policies and save the distribution of
    # the final capitals
    returns1 = evaluate_policy(model, π1)
    returns2 = evaluate_policy(model, π2)
    returns3 = evaluate_policy(model, π3)
    returns4 = evaluate_policy(model, π4)
    save_bar_data(returns1, returns2,
    returns3,returns4)
 
end

main()

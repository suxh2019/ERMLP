"""
code for the paper:Xihong Su, Marek Petrik, Julien Grand-Clément.
Risk-averse Total-reward MDPs with ERM and EVaR. The 39th Annual AAAI 
Conference on Artificial Intelligence(AAAI2025)

Plot for the figure "Distribution of the final capital for EVaR optimal policies"
"""

using Plots


"""
 The distribution α2,α4,α7,α9 are from "data_bar.csv"
"""
function main()
    pgfplotsx()
    capitals = ["-1", "0", "1", "2", "3", "4", "5", "6", "7"]
    α9 = [0.19042857142857142,0,0,0,0,0,0,0,0.8095714285714286]
    α7 = [0.11985714285714286,0,0,0,0,0,0,0,0.8801428571428571]
    α4 = [0,0,0.2591428571428571,0,0,0,0,0,0.7408571428571429]
    α2 = [0,0,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285,
          0.14285714285714285,0.14285714285714285,0.14285714285714285]
    p1 = scatter(capitals, α9, label = "α = 0.9", size=(350,240), legend=:topleft,
                 xlabel="Capital", ylabel = "Probability")
    scatter!(capitals, α7, label = "α = 0.7")
    scatter!(capitals, α4, label = "α = 0.4")
    scatter!(capitals, α2,  label = "α = 0.2")
    plot!(capitals, α9, linestyle=:dash, linecolor=p1.series_list[1][:fillcolor], label=nothing)
    plot!(capitals, α7, linestyle=:dash, linecolor=p1.series_list[2][:fillcolor], label=nothing)
    plot!(capitals, α4, linestyle=:dash, linecolor=p1.series_list[3][:fillcolor], label=nothing)
    plot!(capitals, α2, linestyle=:dash, linecolor=p1.series_list[4][:fillcolor], label=nothing)
    p1
end

p = main()

savefig(p,"final_capital_distri.pdf")


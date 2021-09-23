include("EulerPDMP.jl")

# Setting up the experiment
dim = 50
μ = zeros(dim)
Σ =  zeros(dim,dim) + I
Σ_inv = inv(Σ)
n_exp = 50
T = 10^3
freq = 1.0
times = [freq*i for i = 0:Integer(ceil(T/freq))]

powers = [i for i=1:5]
δ = [10.0^(-powers[i]) for i=1:length(powers)]

## Run the simulation for the Zig-Zag sampler

l1_distance, l2_distance = run_coupled_discrete_continuous_ZigZag_Gauss_exp(μ, Σ_inv, T, δ, n_exp,freq)
# If you want to save or load the results in a tensor uncomment the following lines
# using DataFrames, JLD
# save("l1dist_coupled_zzs_dim50_stdgauss_nexp50_7_21.jld", "data", l1_distance)
# l1_distance = load("l1dist_coupled_zzs_dim50_stdgauss_nexp50_7_21.jld")["data"]  # to load the data if needed
# # save("l2dist_coupled_zzs_dim50_stdgauss_nexp50_final.jld", "data", l2_distance)
# # l2_distance = load("l2dist_coupled_zzs_dim50_stdgauss_nexp50.jld")["data"]

# Make plots for the ℓ^1 distance
for i = 2:length(δ)
    mean_l1 = mean(l1_distance[i,:,:]; dims = 1)
    mean_l1[1] = 10^(-8)  # need this otherwise -Inf at t=0 and it does not work
    var_l1 = var(l1_distance[i,:,:]; dims = 1)
    cur = 1/(10^(i))
    pwr = powers[i]
    if i == 2
        plot(times, mean_l1', label="\$\\delta = 10^{-$pwr} \$",
        legend=:bottomright,
        xlabel = L"\textrm{time}",
        ylims = (10^(-1.5),10^2.3),
        yaxis=:log,
        ylabel=L"\ell^1\textrm{- distance}",
        xtickfontsize=12,
        ytickfontsize=12,
        xguidefontsize=12,
        yguidefontsize=12,
        legendfontsize=12,
        yticks = [10^(-1),10^0,10^1,10^2]
        )
    else
        display(plot!(times,mean_l1',
        yaxis=:log,
        label="\$\\delta = 10^{-$pwr} \$"))
    end
end
savefig("l1dist_coupled_zzs_dim50_stdgauss_nexp50_30aug.pdf")

# Make plots for the ℓ^2 distance
for i = 1:length(δ)
    mean_l2 = mean(l2_distance[i,:,:]; dims = 1)
    mean_l2[1] = 10^(-8)
    var_l2= var(l2_distance[i,:,:]; dims = 1)
    cur = 1/(10^(i))
    pwr = powers[i]
    if i == 1
        plot(time, mean_l2', label="\$\\delta = 10^{-$pwr} \$",
        legend=:bottomright,
        xlabel = L"\textrm{time}",
        # ylims = (10^(-1.7),10^1.5),
        yaxis=:log,
        ylabel=L"\ell^2\textrm{- distance}")
    else
        display(plot!(time,mean_l2',
        yaxis=:log,
        label="\$\\delta = 10^{-$pwr} \$"))
    end
end
savefig("l2dist_coupled_zzs_dim50_stdgauss_nexp50.pdf")

## Run the simulation for the Bouncy Particle sampler

l1_distance, l2_distance = run_coupled_discrete_continuous_BPS_Gauss_exp(μ, Σ_inv, T, δ, n_exp,freq,refresh_rate)

# save("l1dist_coupled_bps_dim50_stdgauss_nexp50_final.jld", "data", l1_distance)
# l1_distance = load("l1dist_coupled_bps_dim50_stdgauss_nexp50_final.jld")["data"]  # to load the data if needed
# save("l2dist_coupled_bps_dim50_stdgauss_nexp50_final.jld", "data", l2_distance)
# # l2_distance = load("l2dist_coupled_zzs_dim50_stdgauss_nexp50.jld")["data"]

# l^1 distance
for i = 2:length(δ)
    mean_l1 = mean(l1_distance[i,:,:]; dims = 1)
    mean_l1[1] = 10^(-8)  # need this otherwise -Inf at t=0 and it does not work
    var_l1 = var(l1_distance[i,:,:]; dims = 1)
    cur = 1/(10^(i))
    pwr = powers[i]
    if i == 2
        plot(times, mean_l1', label="\$\\delta = 10^{-$pwr} \$",
        legend=:bottomright,
        xtickfontsize=12,
        ytickfontsize=12,
        xguidefontsize=12,
        yguidefontsize=12,
        legendfontsize=12,
        xlabel = L"\textrm{time}",
        ylims = (10^(-2.5),10^2),
        yaxis=:log,
        ylabel=L"\ell^1\textrm{- distance}")
    else
        display(plot!(times,mean_l1',
        yaxis=:log,
        label="\$\\delta = 10^{-$pwr} \$"))
    end
end
savefig("l1dist_coupled_bps_dim50_stdgauss_nexp50.pdf")

# l^2 distance
for i = 1:length(δ)
    mean_l2 = mean(l2_distance[i,:,:]; dims = 1)
    mean_l2[1] = 10^(-8)
    var_l2= var(l2_distance[i,:,:]; dims = 1)
    cur = 1/(10^(i))
    pwr = powers[i]
    if i == 1
        plot(time, mean_l2', label="\$\\delta = 10^{-$pwr} \$",
        legend=:bottomright,
        xlabel = L"\textrm{time}",
        ylims = (10^(-4.7),10^1.1),
        yaxis=:log,
        ylabel=L"\ell^2\textrm{- distance}")
        #ylabel = "\$\\ell^1\$")
    else
        display(plot!(time,mean_l2',
        yaxis=:log,
        label="\$\\delta = 10^{-$pwr} \$"))
    end
end
savefig("l2dist_coupled_bps_dim50_stdgauss_nexp50_refr1.pdf")

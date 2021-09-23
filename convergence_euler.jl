include("EulerPDMP.jl")

## EXPERIMENT SETUP: compare accuracy of discretised ZZS to cts ZZS for a gaussian target
dim = 25
μ = zeros(dim)
Σ =  zeros(dim,dim) + I
Σ_inv = inv(Σ)
U(x)    = 0.5 * ((x-μ)') * Σ_inv * (x-μ)
∇U(x)   = Σ_inv*(x-μ)
n_exp = 10^5
powers = [i for i=1:3]
δ = [10.0^(-powers[i]) for i=1:length(powers)]
∂E(i,x) = dot(Σ_inv[i,:],(x-μ))

## Run experiments for the ZZS

ΔT = 5
n_time_steps = 5
time_grid = [i*ΔT  for i=0:n_time_steps]
# Define the Lyapunov function
δ_lyap = 0.1
α_lyap = 0.5
ϕ(s)  =  0.5 * sign(s) * log(1+δ_lyap*abs(s))
V(x,v) = exp(α_lyap * U(x) + sum(ϕ.(v.*∇U(x))))

error_mean_cts,error_rad_cts,moment_lyap_cts,error_mean_euler,error_rad_euler,moment_lyap_euler = convergence_eulerZZS(∇U, V, Σ_inv, μ, Σ, ΔT,n_time_steps, δ, n_exp)

# Plot moment of Lyapunov function
plot(time_grid,moment_lyap_cts[1:end], yaxis=:log, line=:solid, markershape=:circle,
    label="ZZS",xlabel = L"\textrm{time}",
    legend=:topright,
    ylims = (10^(3.5),10^(11.8)),
    ylabel=L"\textrm{moment}\,\,\, \textrm{estimates}\,\,\, \textrm{of }\,\,\, \overline{G}",
    xtickfontsize=12,
    ytickfontsize=12,
    xguidefontsize=12,
    yguidefontsize=12,
    legendfontsize=12,
    xticks = 0:ΔT:(n_time_steps*ΔT),
    yticks = [10^4,10^6,10^8,10^10])
for i = 1:length(δ)
    display(plot!(time_grid,moment_lyap_euler[i,1:end],
            line=:solid, markershape=:circle,label="\$\\delta = 10^{-$i} \$"))
end

savefig("gauss_lyap_moments.pdf")

# Plot of error for the mean
plot(time_grid,error_mean_cts[1:end], yaxis=:log, line=:solid, markershape=:circle,
    label="ZZS",xlabel = L"\textrm{time}",
    #legend=:bottomright,
    ylims = (10^(-3.5),10^0),
    ylabel=L"\textrm{error}\,\,\, \textrm{in}\,\,\,\textrm{the}\,\,\, \textrm{first}\,\,\, \textrm{component}\,\,\,\textrm{of} \,\,\, \mu",
    xticks = 0:ΔT:(n_time_steps*ΔT),
    yticks = [10^(-3),10^(-2),10^(-1),10^0],
    xtickfontsize=12,
    ytickfontsize=12,
    xguidefontsize=12,
    yguidefontsize=12,
    legendfontsize=12,)
for i = 1:length(δ)
    display(plot!(time_grid,error_mean_euler[i,1:end],
            line=:solid, markershape=:circle,label="\$\\delta = 10^{-$i} \$"))
end
savefig("gauss_zzs_meanerror.pdf")

# Plot of error for the radius
plot(time_grid,error_rad_cts[1:end], yaxis=:log, line=:solid, markershape=:circle,
    label="ZZS",xlabel = L"\textrm{time}",
    legend=:topright,
    ylims = (10^(-2.2),10^3.2),
    ylabel=L"\textrm{error}\,\,\,\textrm{for}\,\,\, \textrm{for}\,\,\, \textrm{the}\,\,\, \textrm{radius}\,\,\, \textrm{statistic}",
    xticks = 0:ΔT:(n_time_steps*ΔT),
    yticks = [10^(-2),10^(-1),10^0,10^1,10^2],
    xtickfontsize=12,
    ytickfontsize=12,
    xguidefontsize=12,
    yguidefontsize=12,
    legendfontsize=12,)
for i = 1:length(δ)
    display(plot!(time_grid,error_rad_euler[i,1:end],
            line=:solid, markershape=:circle,label="\$\\delta = 10^{-$i} \$"))
end
savefig("gauss__zzs_radiuserror.pdf")


## Same, but for BPS
ΔT = 20.0
n_time_steps = 5
time_grid = [i*ΔT  for i=0:n_time_steps]
# Define the Lyapunov function
λ(x,v) = max(0,dot(v,∇E(x)))
V(x,v) = exp(0.5 * U(x))/((λ(x,-v)+λ_refr)^0.5)

error_mean_cts,error_rad_cts,moment_lyap_cts,
    error_mean_euler,error_rad_euler,
    moment_lyap_euler = convergence_eulerBPS(∇E, V, λ_refr, Σ_inv, μ,
                Σ, ΔT, n_time_steps, δ, n_exp)


# Plot moment of Lyapunov function
plot(time_grid,moment_lyap_cts[1:end], yaxis=:log, line=:solid, markershape=:circle,
    label="BPS",xlabel = L"\textrm{time}",
    legend=:topright,
    ylims = (10^(3.8),10^(6.2)),
    #ylabel=LaTeXString("moment estimates of $\overline{G}$"),
    ylabel=L"\textrm{moment}\,\,\, \textrm{estimates}\,\,\, \textrm{of }\,\,\, \overline{G}",
    xtickfontsize=12,
    ytickfontsize=12,
    xguidefontsize=12,
    yguidefontsize=12,
    legendfontsize=12,
    xticks = 0:ΔT:(n_time_steps*ΔT),
    yticks = [10^4,10^4.5,10^5,10^5.5,10^6])
for i = 1:length(δ)
    display(plot!(time_grid,moment_lyap_euler[i,1:end],
            line=:solid, markershape=:circle,label="\$\\delta = 10^{-$i} \$"))
end

savefig("bps_gauss_lyap_moments.pdf")

# Plot of error for the mean
plot(time_grid,error_mean_cts[1:end], yaxis=:log, line=:solid, markershape=:circle,
    label="BPS",xlabel = L"\textrm{time}",
    #legend=:bottomright,
    ylims = (10^(-2),10^0),
    ylabel=L"\textrm{error}\,\,\, \textrm{in}\,\,\,\textrm{the}\,\,\, \textrm{first}\,\,\, \textrm{component}\,\,\,\textrm{of} \,\,\, \mu",
    xticks = 0:ΔT:(n_time_steps*ΔT),
    xtickfontsize=12,
    ytickfontsize=12,
    xguidefontsize=12,
    yguidefontsize=12,
    legendfontsize=12,)
for i = 1:length(δ)
    display(plot!(time_grid,error_mean_euler[i,1:end],
            line=:solid, markershape=:circle,label="\$\\delta = 10^{-$i} \$"))
end

savefig("gauss_bps_meanerror.pdf")

# Plot of error for the radius
plot(time_grid,error_rad_cts[1:end], yaxis=:log, line=:solid, markershape=:circle,
    label="BPS",xlabel = L"\textrm{time}",
    legend=:topright,
    ylims = (10^(0),10^1),
    ylabel=L"\textrm{error}\,\,\,\textrm{for}\,\,\, \textrm{for}\,\,\, \textrm{the}\,\,\, \textrm{radius}\,\,\, \textrm{statistic}",
    xticks = 0:ΔT:(n_time_steps*ΔT),
    yticks = [10^0,10^0.5,10^1],
    xtickfontsize=12,
    ytickfontsize=12,
    xguidefontsize=12,
    yguidefontsize=12,
    legendfontsize=12,)
for i = 1:length(δ)
    display(plot!(time_grid,error_rad_euler[i,1:end],
            line=:solid, markershape=:circle,label="\$\\delta = 10^{-$i} \$"))
end
savefig("gauss_bps_radiuserror.pdf")

using Statistics, LinearAlgebra, Compat, Plots, LaTeXStrings, StatsBase, StatsPlots
using BenchmarkTools, Optim

struct skeleton
  position::Array{Float64,1}
  velocity::Array{Float64,1}
  time::Float64
end

function getPosition(skele::Array{skeleton,1}; i_start::Integer=1, i_end::Integer=0, want_array::Bool=false)
  if i_end == 0
    i_end = length(skele)
  end
  n_samples = i_end - i_start + 1
  dim = size(skele[1].position,1)
  if want_array
    position = Array{Float64,2}(undef,dim,n_samples)
    for i = i_start:i_end
      position[:,i] = skele[i].position
    end
  else
    position = Vector{Vector{Float64}}(undef,0)
    for i = i_start:i_end
      push!(position,skele[i].position)
    end
  end
  return position
end

function getVelocity(skele::Array{skeleton,1}; i_start::Integer=1, i_end::Integer=0,want_array::Bool=false)
  if i_end ==0
    i_end = length(skele)
  end
  n_samples = i_end - i_start + 1
  dim = size(skele[1].position,1)
  if want_array
    velocity = Array{Int64,2}(undef,dim,n_samples)
    for i = i_start:i_end
      velocity[:,i] = skele[i].velocity
    end
  else
    velocity = Vector{Vector{Float64}}(undef,0)
    for i = i_start:i_end
      push!(velocity,skele[i].velocity)
    end
  end
  return velocity
end

function getVelocity_bps(skele::Array{skeleton,1}; i_start::Integer=1, i_end::Integer=0,want_array::Bool=false)
  if i_end ==0
    i_end = length(skele)
  end
  n_samples = i_end - i_start + 1
  dim = size(skele[1].position,1)
  if want_array
    velocity = Array{Float64,2}(undef,dim,n_samples)
    for i = i_start:i_end
      velocity[:,i] = skele[i].velocity
    end
  else
    velocity = Vector{Vector{Float64}}(undef,0)
    for i = i_start:i_end
      push!(velocity,skele[i].velocity)
    end
  end
  return velocity
end

function getTime(skele::Array{skeleton,1}; i_start::Integer=1, i_end::Integer=0)
  if i_end ==0
    i_end = length(skele)
  end
  #time = []
  time =  Vector{Float64}(undef, 0)
  for i = i_start:1:i_end
    push!(time, skele[i].time)
  end
  return time
end

function discretise(skel_chain::AbstractArray, ??t::Float64, t_fin::Real, d::Integer)
  # Discretises the process described in skel_chain with step ??t for the interval
  # starting with the first time in skel_chain up to time t_fin (or the nearest
  # time point in the grid). The initial position is the first position in skel_chain.
  # The output is the process at discrete times, where the
  # first point in skel_chain is not considered.
  time = skel_chain[1].time
  i_skel = 1
  dim_skel = length(skel_chain)
  dim_temp = Int(round((t_fin-time)/??t))
  temp = Array{Float64,2}(undef,d, dim_temp+1)
  temp[:,1] = skel_chain[1].position #this must be at time 0.0 or n*t_adaps
  for k = 1:dim_temp
    time += ??t
    if i_skel==dim_skel || time <= skel_chain[i_skel+1].time
      temp[:,k+1]= temp[:,k] + skel_chain[i_skel].velocity*??t;
    else
      while ((i_skel+1) <= dim_skel) && (time > skel_chain[i_skel+1].time)
        i_skel+=1;
      end
      t_left = time - skel_chain[i_skel].time;
      if (t_left > ??t)
        temp[:,k+1]= temp[:,k] + skel_chain[i_skel].velocity*??t;
      else
        temp[:,k+1]= skel_chain[i_skel].position + skel_chain[i_skel].velocity*t_left;
      end
    end
  end
  return temp[:,2:end]
end

function switchingtime(a::Float64,b::Float64,u::Float64=rand())
# generate switching time for rate of the form max(0, a + b s) + c
# under the assumptions that b > 0, c > 0
  if (b > 0)
    if (a < 0)
      return -a/b + switchingtime(0.0, b, u);
    else # a >= 0
      return -a/b + sqrt(a^2/b^2 - 2 * log(1-u)/b);
    end
  elseif (b == 0) # degenerate case
    if (a < 0)
      return Inf;
    else # a >= 0
      return -log(1-u)/a;
    end
  else # b <= 0
    if (a <= 0)
      return Inf;
    else # a > 0
      y = -log(1-u); t1=-a/b;
      if (y >= a * t1 + b *t1^2/2)
        return Inf;
      else
        return -a/b - sqrt(a^2/b^2 + 2 * y /b);
      end
    end
  end
end

function ZigZag(???E::Function, Q::Matrix{Float64}, T::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
    # E_partial_derivative(i,x) is the i-th partial derivative of the potential E, evaluated in x
    # Q is a symmetric matrix with nonnegative entries such that |(???^2 E(x))_{ij}| <= Q_{ij} for all x, i, j
    # T is time horizon

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        # x_init = zeros(dim)
        x_init = randn(dim)
        v_init = rand((-1,1), dim)
    end

    b = [norm(Q[:,i]) for i=1:dim];
    b = sqrt(dim)*b;
    # b = Q * ones(dim);

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))
    rejected_switches = 0;
    accepted_switches = 0;
    initial_gradient = [???E(i,x) for i in 1:dim];
    a = v .* initial_gradient

    ??t_proposed_switches = switchingtime.(a,b)
    if (excess_rate == 0.0)
        ??t_excess = Inf
    else
        ??t_excess = -log(rand())/(dim*excess_rate)
    end

    while (!finished)
        i = argmin(??t_proposed_switches) # O(d)
        ??t_switch_proposed = ??t_proposed_switches[i]
        ??t = min(??t_switch_proposed,??t_excess);
        if t + ??t > T
            ??t = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * ??t; # O(d)
        t = t + ??t;
        a = a + b * ??t; # O(d)

        if (!finished && ??t_switch_proposed < ??t_excess)
            switch_rate = v[i] * ???E(i,x)
            proposedSwitchIntensity = a[i]
            if proposedSwitchIntensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v[i] = -v[i]
                a[i] = -switch_rate
                updateSkeleton = true
                accepted_switches += 1
            else
                a[i] = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            ??t_excess = ??t_excess - ??t_switch_proposed
            ??t_proposed_switches = ??t_proposed_switches .- ??t_switch_proposed
            ??t_proposed_switches[i] = switchingtime(a[i],b[i])
        elseif !finished
            # so we switch due to excess switching rate
            updateSkeleton = true
            i = rand(1:dim)
            v[i] = -v[i]
            a[i] = v[i] * ???E(i,x)

            # update upcoming event times
            ??t_proposed_switches = ??t_proposed_switches .- ??t_excess
            ??t_excess = -log(rand())/(dim*excess_rate);
        end

        if updateSkeleton
            push!(skel_chain,skeleton(x,v,t))
            # push!(x_skeleton, x)
            # push!(v_skeleton, v)
            # push!(t_skeleton, t)
            updateSkeleton = false
        end

    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    # return (t_skeleton, x_skeleton, v_skeleton)

    return skel_chain

end



function flip!(vel::Vector, i::Integer)
  temp = vel[i]
  vel[i] = -temp
end


function reflect(gradient::Vector{Float64}, v::Vector{Float64})

    return v - 2 * (transpose(gradient) * v / dot(gradient,gradient)) * gradient

end

tolerance = 1e-7 # for comparing switching rate and bound

function BPS(???E::Function, Q::Matrix{Float64}, T::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Float64} = Vector{Float64}(undef,0), refresh_rate::Float64 = 1.0)
    # g_E! is the gradient of the energy function E
    # Q is a symmetric matrix such that Q - ???^2 E(x) is positive semidefinite

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = randn(dim)
    end

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;

    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))

    rejected_switches = 0;
    accepted_switches = 0;
    gradient = ???E(x);
    a = transpose(v) * gradient;
    b = transpose(v) * Q * v;

    ??t_switch_proposed = switchingtime(a,b)
    if refresh_rate <= 0.0
        ??t_refresh = Inf
    else
        ??t_refresh = -log(rand())/refresh_rate
    end

    while (!finished)
        ??t = min(??t_switch_proposed,??t_refresh);
        if t + ??t > T
            ??t = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * ??t; # O(d)
        t = t + ??t;
        a = a + b * ??t; # O(d)
        gradient = ???E(x)

        if (!finished && ??t_switch_proposed < ??t_refresh)
            switch_rate = transpose(v) * gradient
            proposedSwitchIntensity = a
            if proposedSwitchIntensity < switch_rate - tolerance
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v = reflect(gradient,v)
                a = -switch_rate
                b = transpose(v) * Q * v
                updateSkeleton = true
                accepted_switches += 1
            else
                a = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update time to refresh
            ??t_refresh = ??t_refresh - ??t_switch_proposed
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = randn(dim)
            a = transpose(v) * gradient
            b = transpose(v) * Q * v

            # update time to refresh
            ??t_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            # push!(x_skeleton, x)
            # push!(v_skeleton, v)
            # push!(t_skeleton, t)
            push!(skel_chain,skeleton(x,v,t))
            updateSkeleton = false
        end
        ??t_switch_proposed = switchingtime(a,b)
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    # return (t_skeleton, x_skeleton, v_skeleton)
    return skel_chain
end

function BPS_unitsphere(???E::Function, Q::Matrix{Float64}, T::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Float64} = Vector{Float64}(undef,0), refresh_rate::Float64 = 1.0)
    # g_E! is the gradient of the energy function E
    # Q is a symmetric matrix such that Q - ???^2 E(x) is positive semidefinite

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = randn(dim)
        v_init = v_init./norm(v_init)
    end

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;

    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))

    rejected_switches = 0;
    accepted_switches = 0;
    gradient = ???E(x);
    a = transpose(v) * gradient;
    b = transpose(v) * Q * v;

    ??t_switch_proposed = switchingtime(a,b)
    if refresh_rate <= 0.0
        ??t_refresh = Inf
    else
        ??t_refresh = -log(rand())/refresh_rate
    end

    while (!finished)
        ??t = min(??t_switch_proposed,??t_refresh);
        if t + ??t > T
            ??t = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * ??t; # O(d)
        t = t + ??t;
        a = a + b * ??t; # O(d)
        gradient = ???E(x)

        if (!finished && ??t_switch_proposed < ??t_refresh)
            switch_rate = transpose(v) * gradient
            proposedSwitchIntensity = a
            if proposedSwitchIntensity < switch_rate - tolerance
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v = reflect(gradient,v)
                a = -switch_rate
                b = transpose(v) * Q * v
                updateSkeleton = true
                accepted_switches += 1
            else
                a = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update time to refresh
            ??t_refresh = ??t_refresh - ??t_switch_proposed
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = randn(dim)
            v = v./norm(v)
            a = transpose(v) * gradient
            b = transpose(v) * Q * v

            # update time to refresh
            ??t_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            # push!(x_skeleton, x)
            # push!(v_skeleton, v)
            # push!(t_skeleton, t)
            push!(skel_chain,skeleton(x,v,t))
            updateSkeleton = false
        end
        ??t_switch_proposed = switchingtime(a,b)
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    # return (t_skeleton, x_skeleton, v_skeleton)
    return skel_chain
end

function EulerZigZag(???E::Function, T::Real, dim::Int64, ??::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)

    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = randn(dim)
        v_init = rand((-1,1), dim)
    end

    n=0;
    x = x_init; v = v_init;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(copy(x),copy(v),0))

    if excess_rate > 0
        println("ERROR: strictly positive excess switching rate currently not supported.")
        error("Excess switching rate currently not supported.")
    end

    while (!finished)
        n+=1
        if n*??>=T
            finished = true
        end
        U = rand(dim)
        ?? = max.(v.*???E(x),0)      #??compute frozen switching rates
        x = x + v * ??     # now move deterministically to new position
        t_switch = -log.(1 .- U)./??
        i = argmin(t_switch)
        if t_switch[i] < ??
            v[i] = - v[i]
        end
        # for i = 1:dim     # check which components of the velocity vector are flipped
        #     if 1-exp(-??*??[i]) > U[i]
        #         v[i] = - v[i]
        #     end
        # end
        push!(skel_chain,skeleton(copy(x),copy(v),n*??))
    end

    return skel_chain

end

function EulerBPS(???E::Function, dim::Integer, ??_refr::Real, T::Real, ??::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Float64} = Vector{Float64}(undef,0))
    # Implementation of the Partially Discrete approximation of BPS
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = randn(dim)
        v_init = randn(dim)
    end

    n=0;
    x = x_init; v = v_init;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(copy(x),copy(v),0))

    while (!finished)
        n+=1
        if n*??>=T
            finished = true
        end
        gradient = ???E(x)
        ?? = max(0,dot(v,gradient))  # compute freezed switching rate
        U = rand(2)
        t_refl = -log(1-U[1])/??
        t_refr = -log(1-U[2])/??_refr
        t_switch = min(t_refl,t_refr)
        if t_switch <= ?? # then there is a switching event
            x = x + v * t_switch
            if t_refl < t_refr  # the particle is reflected
                gradient = ???E(x)
                v = reflect(gradient,v)  # reflection using the new gradient
            else    # velocity refreshment
                v = randn(dim)
            end
            x = x + v * (??-t_switch)
        else
            x = x + v * ??
        end
        push!(skel_chain,skeleton(copy(x),copy(v),n*??))
    end
    return skel_chain
end


function EulerBPS_unitsphere(???E::Function, dim::Integer, ??_refr::Real, T::Real, ??::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
    v_init::Vector{Float64} = Vector{Float64}(undef,0))
    # Implementation of the Partially Discrete approximation of BPS
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = randn(dim)
        v_init = randn(dim)
        v_init = v_init./norm(v_init)
    end

    n=0;
    x = x_init; v = v_init;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(copy(x),copy(v),0))

    while (!finished)
        n+=1
        if n*??>=T
            finished = true
        end
        gradient = ???E(x)
        ?? = max(0,dot(v,gradient))  # compute freezed switching rate
        U = rand(2)
        t_refl = -log(1-U[1])/??
        t_refr = -log(1-U[2])/??_refr
        t_switch = min(t_refl,t_refr)
        if t_switch <= ?? # then there is a switching event
            x = x + v * t_switch
            if t_refl < t_refr  # the particle is reflected
                gradient = ???E(x)
                v = reflect(gradient,v)  # reflection using the new gradient
            else    # velocity refreshment
                v = randn(dim)
                v = v./norm(v)
            end
            x = x + v * (??-t_switch)
        else
            x = x + v * ??
        end
        push!(skel_chain,skeleton(copy(x),copy(v),n*??))
    end
    return skel_chain
end

function compute_errors(chain::Vector{skeleton}, ??::Vector{Float64}, rad::Real)
    positions = getPosition(chain; want_array=true)
    error_??   = abs(mean(positions[1,:])-??[1])
    error_rad = abs(mean(sum(positions.^2; dims = 1))-rad)
    return error_??, error_rad
end

function coupled_discrete_continuous_ZigZag_Gauss_exp(??::Vector, ??_inv::Matrix{Float64}, T::Real, ??::Real, n_exp::Integer, freq::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
    # This function runs n_exp times the coupling of the two processes until time T

    dim = size(??_inv)[1]
    # if (length(x_init) == 0 || length(v_init) == 0)
    #     x_init = zeros(dim)
    #     v_init = rand((-1,1), dim)
    # end

    nr_pts = Integer(ceil(T/freq))
    l1_distance = Array{Float64,2}(undef,n_exp, nr_pts+1)
    l2_distance = Array{Float64,2}(undef,n_exp, nr_pts+1)
    l1_distance[:,1] .= 0
    l2_distance[:,1] .= 0
    n_steps = Integer(ceil(T/??))

    for k = 1:n_exp
        x_init = zeros(dim)
        v_init = rand((-1,1), dim)
        t = 0.0; n=0; counter = 1;
        x = copy(x_init); v = copy(v_init);
        x_euler = copy(x_init); v_euler = copy(v_init);
        finished = false;

        while (!finished)
            n += 1
            U = rand(dim)
            # simulate continuous zig-zag process
            a = v.*(??_inv * (x - ??));
            b = v.*(??_inv * v);
            ??t_proposed_switches = switchingtime.(a,b,U)
            i = argmin(??t_proposed_switches) # O(d)
            ??t = ??t_proposed_switches[i]
            if ??t <= ??  # then at least one velocity flip takes place
                x = x + v * ??t
                v[i] = -v[i]
                t = t + ??t
                # then we have a loop until we get to the end of the interval
                t_remaining = ?? - ??t   #remaining time in this interval
                while t_remaining > 0
                    # compute next switching time
                    a = v.*(??_inv * (x - ??));
                    b = v.*(??_inv * v);
                    ??t_proposed_switches = switchingtime.(a,b)
                    i = argmin(??t_proposed_switches) # O(d)
                    ??t = ??t_proposed_switches[i]
                    if ??t<=t_remaining # then we have another switch
                        x = x + v * ??t
                        v[i]=-v[i]
                        t = t + ??t
                        t_remaining -= ??t
                    else # we got to the end of this time step
                        x = x + v * t_remaining
                        t = n*??
                        t_remaining = -1  # exit the loop
                    end
                end
            else # then no switches for the continuous zigzag in this time interval
                x = x + v*??
                t = n*??
            end
            # simulate Euler-discretised zig-zag process
            ??_euler = max.(v_euler.*??_inv*(x_euler-??),0)
            x_euler = x_euler + v_euler * ?? # this is not affected by velocity switches
            t_switch = -log.(1 .- U)./??_euler
            i = argmin(t_switch)
            if t_switch[i] < ??
                v_euler[i] = - v_euler[i]
            end
            # for i = 1:dim
            #     if U[i] <= 1-exp(-??*??_euler[i])
            #         v_euler[i] = - v_euler[i]
            #     end
            # end
            if t >= counter*freq
                # then update distances
                counter += 1
                l1_distance[k,counter] = norm(x-x_euler,1) + norm(v-v_euler,1)
                l2_distance[k,counter] = norm(x-x_euler,2) + norm(v-v_euler,2)
            end
            if t>=T
                finished = true
            end
        end
    end


    return l1_distance, l2_distance

end

function run_coupled_discrete_continuous_ZigZag_Gauss_exp(??::Vector, ??_inv::Matrix{Float64}, T::Real, ??::Vector, n_exp::Integer, freq::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
  # Run experiments for several values of delta, return a tensor
  nr_pts = Integer(ceil(T/freq))
  l1_distance = Array{Float64,3}(undef,length(??),n_exp, nr_pts+1)
  l2_distance = Array{Float64,3}(undef,length(??),n_exp, nr_pts+1)
  for i = 1:length(??)
      l1_distance[i,:,:], l2_distance[i,:,:] = coupled_discrete_continuous_ZigZag_Gauss_exp(??, ??_inv, T, ??[i], n_exp,freq)
      println("Done with the $i-th value of delta")
  end
  return l1_distance, l2_distance
end


function coupled_discrete_continuous_BPS_Gauss_exp(??::Vector, ??_inv::Matrix{Float64}, T::Real, ??::Real, n_exp::Integer, freq::Real, refresh_rate::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
                                                    v_init::Vector{Int} = Vector{Int}(undef,0))
    # This function runs n_exp times the coupling of the two processes until time T

    dim = size(??_inv)[1]
    nr_pts = Integer(ceil(T/freq))
    l1_distance = Array{Float64,2}(undef,n_exp, nr_pts+1)
    l2_distance = Array{Float64,2}(undef,n_exp, nr_pts+1)
    # l1_distance[:,1] .= 0
    # l2_distance[:,1] .= 0
    n_steps = Integer(ceil(T/??))

    for k = 1:n_exp
        t = 0.0; n=0; counter = 1;
        # x = copy(x_init); v = copy(v_init);
        # x_euler = copy(x_init); v_euler = copy(v_init_e);
        x = randn(dim); v = randn(dim);
        # x_euler = randn(dim); v_euler = randn(dim);
        x_euler = copy(x); v_euler = copy(v);
        l1_distance[k,1] = norm(x-x_euler,1) + norm(v-v_euler,1)
        l2_distance[k,1] = norm(x-x_euler,2) + norm(v-v_euler,2)
        finished = false;
        if refresh_rate <= 0.0
            ??t_refresh = Inf
        else
            ??t_refresh = -log(rand())/refresh_rate
        end

        while (!finished)
            n += 1
            U_1 = rand()

            # simulate Euler-discretised bps
            ??_euler = max(dot(v_euler,??_inv*(x_euler-??)),0)
            if ??_euler == 0
                ??t_reflect_euler = Inf
            else
                ??t_reflect_euler = U_1/??_euler
            end
            if min(??t_reflect_euler,??t_refresh) < ??
                if ??t_reflect_euler < ??t_refresh
                    gradient = ??_inv*(x_euler-??)
                    x_euler = x_euler + v_euler * ??
                    v_euler = reflect(gradient,v_euler)
                else     # then there is a velocity refreshment
                    x_euler = x_euler + v_euler * ??
                    v_euler = randn(dim)
                end
            else    # update only the position of the process
                x_euler = x_euler + v_euler * ??
            end

            # simulate continuous bps
            a = dot(v,x-??)
            b = dot(v,??_inv*v)
            ??t_reflect = switchingtime(a,b,U_1)
            ??t = min(??t_reflect,??t_refresh)
            if ??t <= ??  # then at least one reflection takes place
                x = x + v * ??t   # move to the right position
                if ??t_reflect <= ??t_refresh
                    gradient = ??_inv*(x-??)
                    v = reflect(gradient,v)
                    ??t_refresh -= ??t_reflect
                else   # refresh the velocity
                    if (??t_refresh < ??t_reflect_euler)  && (??t_refresh < ??)   # then the approximation had a refreshment in this time step --> copy the velocity
                        v = copy(v_euler)
                    else
                        v = randn(dim)
                    end
                    ??t_refresh = -log(rand())/refresh_rate
                end
                t = t + ??t
                # then we have a loop until we get to the end of the interval
                t_remaining = ?? - ??t   #remaining time in this interval
                while t_remaining > 0
                    # compute next switching time
                    a = dot(v,x-??)
                    b = dot(v,??_inv*v)
                    ??t_reflect = switchingtime(a,b)
                    ??t = min(??t_reflect,??t_refresh)
                    if ??t<=t_remaining # then we have another switch
                        if ??t_reflect <= ??t_refresh
                            gradient = ??_inv*(x-??)
                            v = reflect(gradient,v)
                            ??t_refresh -= ??t_reflect
                        else
                            v = randn(dim)
                            ??t_refresh = -log(rand())/refresh_rate
                        end
                        t = t + ??t
                        t_remaining -= ??t
                    else # we got to the end of this time step
                        x = x + v * t_remaining
                        t = n*??
                        t_remaining = -1  # exit the loop
                    end
                end
            else # then no switches for the continuous bps in this time interval
                x = x + v*??
                t = n*??
            end
            if refresh_rate <= 0.0
                ??t_refresh = Inf
            else
                ??t_refresh = -log(rand())/refresh_rate
            end

            if t >= counter*freq
                # then update distances
                counter += 1
                l1_distance[k,counter] = norm(x-x_euler,1) + norm(v-v_euler,1)
                l2_distance[k,counter] = norm(x-x_euler,2) + norm(v-v_euler,2)
            end
            if t>=T
                finished = true
            end
        end
    end

    return l1_distance, l2_distance

end

function run_coupled_discrete_continuous_BPS_Gauss_exp(??::Vector, ??_inv::Matrix{Float64}, T::Real, ??::Vector, n_exp::Integer, freq::Real, refresh_rate::Real, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
  # Run experiments for several values of delta, return a tensor
  nr_pts = Integer(ceil(T/freq))
  l1_distance = Array{Float64,3}(undef,length(??),n_exp, nr_pts+1)
  l2_distance = Array{Float64,3}(undef,length(??),n_exp, nr_pts+1)
  for i = 1:length(??)
      l1_distance[i,:,:], l2_distance[i,:,:] = coupled_discrete_continuous_BPS_Gauss_exp(??,??_inv,T,??[i], n_exp,freq,refresh_rate)
  end
  return l1_distance, l2_distance
end

function convergence_eulerZZS(???E::Function, V::Function, ??_inv::Matrix{Float64}, ??::Vector{Float64},
            ??::Matrix{Float64}, ??T::Real, n_time_steps::Integer, ??::Vector{Float64}, n_exp::Integer,
            x_init::Vector{Float64} = Vector{Float64}(undef,0),
            v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
  # Estimate two things: mean and radius.
  dim = size(??)[1]
  error_mean_euler = zeros(length(??),n_time_steps+1)
  error_rad_euler = zeros(length(??),n_time_steps+1)
  moment_lyap_euler = zeros(length(??),n_time_steps+1)
  last_pos_cts = zeros(n_time_steps+1,n_exp);
  radius_cts=Matrix{Float64}(undef,n_time_steps+1,n_exp)
  lyap_cts  = Matrix{Float64}(undef,n_time_steps+1,n_exp)

  mean_radius_stat = sum(??.^2) + sum(diag(??))
  for j = 1:length(??)
    println("Starting with $j-th value of delta")
      last_pos_euler = zeros(n_time_steps+1,n_exp)
      radius_euler   = Matrix{Float64}(undef,n_time_steps+1,n_exp)
      lyap_euler     = Matrix{Float64}(undef,n_time_steps+1,n_exp)
      for k = 1:n_exp
          x_init = rand(dim)+randn(dim)
          v_init = rand((-1,1), dim)
          last_pos_euler[1,k]   =   x_init[1]
          radius_euler[1,k]     =   sum(x_init.^2)
          lyap_euler[1,k]       =   V(x_init,v_init)
          for l = 1:n_time_steps
              euler_chain = EulerZigZag(???E, ??T, dim, ??[j], x_init, v_init, excess_rate)
              # check = (norm(getPosition(euler_chain; want_array = true)[:,1]-x_init) > 10^(-5))
              # println("$check")
              x_init = getPosition(euler_chain; want_array = true)[:,end]
              v_init = getVelocity(euler_chain; want_array = true)[:,end]
              last_pos_euler[l+1,k] = x_init[1]
              radius_euler[l+1,k] = sum(x_init.^2)
              lyap_euler[l+1,k] = V(x_init,v_init)
          end
      end
      moment_lyap_euler[j,:]    =   [mean(lyap_euler[m,:])  for m=1:(n_time_steps +1)]
      error_mean_euler[j,:]     =   [abs(mean(last_pos_euler[m,:])-??[1])   for m=1:(n_time_steps +1)]
      error_rad_euler[j,:]      =   [abs(mean(radius_euler[m,:]) - mean_radius_stat)  for m=1:(n_time_steps +1)]
      # var_error_mean_euler[j,:] =   [sum(abs.(last_pos_euler[m,:,:]-??))   for m=1:(n_time_steps +1)]
      println("Done with the $j-th value of delta")
  end
  println("Done with the experiments for the discretised process.")
  for i = 1:n_exp
      x_init = rand(dim)+randn(dim)
      v_init = rand((-1,1), dim)
      last_pos_cts[1,i] =   x_init[1]
      radius_cts[1,i]     =   sum(x_init.^2)
      lyap_cts[1,i]       =   V(x_init,v_init)
      for l = 2:(n_time_steps+1)
          skel_chain  = ZigZag(???E, ??_inv, ??T, x_init, v_init, excess_rate)
          x_init = getPosition(skel_chain; want_array = true)[:,end]
          v_init = getVelocity(skel_chain; want_array = true)[:,end]
          last_pos_cts[l,i] = x_init[1]
          radius_cts[l,i]   = sum(x_init.^2)
          lyap_cts[l,i]     = V(x_init,v_init)
      end
  end
  moment_lyap_cts =   [mean(lyap_cts[m,:])  for m=1:(n_time_steps +1)]
  error_mean_cts  =   [abs(mean(last_pos_cts[m,:])-??[1])  for m=1:(n_time_steps +1)]
  error_rad_cts   =   [abs(mean(radius_cts[m,:])- mean_radius_stat)    for m=1:(n_time_steps +1)]

  return error_mean_cts,error_rad_cts,moment_lyap_cts,error_mean_euler,error_rad_euler,moment_lyap_euler
end

function convergence_eulerBPS(???E::Function, V::Function, ??_refr::Real, ??_inv::Matrix{Float64}, ??::Vector{Float64},
            ??::Matrix{Float64}, ??T::Real, n_time_steps::Integer, ??::Vector{Float64}, n_exp::Integer;
            x_init::Vector{Float64} = Vector{Float64}(undef,0),
            v_init::Vector{Int} = Vector{Int}(undef,0))
  # Estimate three things: errors for the mean and radius,
  # as well as the moment of the Lyapunov function V.
  dim = size(??)[1]
  error_mean_euler  = zeros(length(??),n_time_steps+1)
  error_rad_euler   = zeros(length(??),n_time_steps+1)
  moment_lyap_euler = zeros(length(??),n_time_steps+1)
  last_pos_cts      = zeros(n_time_steps+1,n_exp);
  radius_cts        = Matrix{Float64}(undef,n_time_steps+1,n_exp)
  lyap_cts          = Matrix{Float64}(undef,n_time_steps+1,n_exp)

  mean_radius_stat = sum(??.^2) + sum(diag(??))

  for j = 1:length(??)
      println("Starting with $j-th value of delta")
      last_pos_euler = zeros(n_time_steps+1,n_exp)
      radius_euler   = Matrix{Float64}(undef,n_time_steps+1,n_exp)
      lyap_euler     = Matrix{Float64}(undef,n_time_steps+1,n_exp)
      for k = 1:n_exp
          x_init = rand(dim)+randn(dim)
          v_init = randn(dim)
          v_init = v_init./norm(v_init)
          last_pos_euler[1,k]   =   x_init[1]
          radius_euler[1,k]     =   sum(x_init.^2)
          lyap_euler[1,k]       =   V(x_init,v_init)
          for l = 1:n_time_steps
              euler_chain = EulerBPS_unitsphere(???E, dim, ??_refr, ??T, ??[j], x_init, v_init)
              x_init = euler_chain[end].position
              v_init = euler_chain[end].velocity
              # x_init = getPosition(euler_chain; want_array = true)[:,end]
              # v_init = getVelocity_bps(euler_chain; want_array = true)[:,end]
              last_pos_euler[l+1,k] = x_init[1]
              radius_euler[l+1,k] = sum(x_init.^2)
              lyap_euler[l+1,k] = V(x_init,v_init)
          end
      end
      moment_lyap_euler[j,:]    =   [mean(lyap_euler[m,:])  for m=1:(n_time_steps +1)]
      error_mean_euler[j,:]     =   [abs(mean(last_pos_euler[m,:])-??[1])   for m=1:(n_time_steps +1)]
      error_rad_euler[j,:]      =   [abs(mean(radius_euler[m,:]) - mean_radius_stat)  for m=1:(n_time_steps +1)]
      # var_error_mean_euler[j,:] =   [sum(abs.(last_pos_euler[m,:,:]-??))   for m=1:(n_time_steps +1)]
      println("Done with the $j-th value of delta")
  end
  println("Done with the experiments for the discretised process.")
  for i = 1:n_exp
      x_init = rand(dim)+randn(dim)
      v_init = randn(dim)
      v_init = v_init./norm(v_init)
      last_pos_cts[1,i] =   x_init[1]
      radius_cts[1,i]     =   sum(x_init.^2)
      lyap_cts[1,i]       =   V(x_init,v_init)
      for l = 2:(n_time_steps+1)
          skel_chain  = BPS_unitsphere(???E, ??_inv, ??T, x_init, v_init, ??_refr)
          x_init = getPosition(skel_chain; want_array = true)[:,end]
          v_init = getVelocity_bps(skel_chain; want_array = true)[:,end]
          last_pos_cts[l,i] = x_init[1]
          radius_cts[l,i]   = sum(x_init.^2)
          lyap_cts[l,i]     = V(x_init,v_init)
      end
  end
  moment_lyap_cts =   [mean(lyap_cts[m,:])  for m=1:(n_time_steps +1)]
  error_mean_cts  =   [abs(mean(last_pos_cts[m,:])-??[1])  for m=1:(n_time_steps +1)]
  error_rad_cts   =   [abs(mean(radius_cts[m,:])- mean_radius_stat)    for m=1:(n_time_steps +1)]

  return error_mean_cts,error_rad_cts,moment_lyap_cts,error_mean_euler,error_rad_euler,moment_lyap_euler
end

using Distributed
using DistributedArrays
using DistributedArrays.SPMD
@everywhere using Distributed
@everywhere using DistributedArrays
@everywhere using DistributedArrays.SPMD



# Derivadas

"""
get_x(i,dx)
gives the values of x[i], starting with 0
"""
get_x(i,dx) = (i-1)*dx 

"""
get_p(j,dp,Np)
gives the value of p symmetric around zero
"""
get_p(j,dp,Np) = (j - (Np+1)÷ 2) * dp 


include("derivs.jl")

################################ CONSTRAINTS #################################

"""
The following function evaluates the electric field on a uniform grid from the electric potential.
    E = -grad phi

    // Calculate electric field from potential
"""
function get_E_from_ϕ!(ϕ, E, dx)
      J = length(E)
      for j in 2:J-1
        E[j] = (ϕ[j-1] - ϕ[j+1]) / 2. / dx
      end
      E[1] = (ϕ[J] - ϕ[2]) / 2. / dx;
      E[J] = (ϕ[J-1] - ϕ[1]) / 2. / dx;
      return E[:]
end

""" The following routine solves Poisson's equation in 1-D to find the instantaneous electric potential on a uniform grid.

Solves 1-d Poisson equation:
    d^{phi} / dx^2 = -4*pi *rho   for  0 <= x <= L
 Periodic boundary conditions:
    u(x + L) = u(x),  v(x + L) = v(x)
 Arrays u and v assumed to be of length J.
 Now, jth grid point corresponds to
    x_j = j dx  for j = 0,J-1
 where dx = L / J. L / (J-1) in Julia
 Also,
    kappa = 2 pi / L
"""
function get_ϕ!(ϕ, ρ, κ)
  #V = fill(0.0+im*0.,J) 
  #U = fill(0.0+im*0.,J÷2+1) 
  J = length(ρ)
  # Fourier transform source term
  V = rfft(ρ)

  # Calculate Fourier transform of u

  V[1] =  0.
  for j in  2:(J÷2+1)
    V[j] =  V[j] / (j-1)^2 / κ^2
  end

  # Inverse Fourier transform to obtain u
  ϕ[:] = irfft(V,J)
end

"""
Takes out the mass of the grid function so that the sum is now null
"""
function filter_constant!(E)
  J = length(E)
  V = rfft(E)
  V[1] = 0.0 #extract the first component
  E[:] = irfft(V,J)
end

##################################### CURRENT AND DENSITY #################################

function get_current!(u,S,par)
    Nx, dx, Np, dp, v, m, e = par
    F = reshape(u,(Nx,Np+1))
    @threads  for i in 1:Nx
        @inbounds S[i] = 0
        for j in 1:Np
            #p = get_p(j, dp, Np)/m
            @inbounds    S[i] += e * F[i,j]* v[j] * dp
        end
    end
    return S
end

function get_density!(u,ρ,par)
    Nx, dx, Np, dp, m, e = par
    F = reshape(u,(Nx,Np+1))
    for i in 1:Nx
        ρ[i] = 0
        for j in 1:Np  
            ρ[i] += e * F[i,j] * dp
        end
    end
    return ρ
end

function get_total_density!(ρ,par)
    Nx, dx = par
    n0 = 0.0
    for i in 1:Nx
        n0 += ρ[i] * dx
    end
    return n0 
end

function get_K_energy!(u,E_K,par)
    Nx, dx, Np, dp = par
    F = reshape(u,(Nx,Np+1))
    for i in 1:Nx
        E_K[i] = 0.0
        for j in 1:Np  
            p = get_p(j, dp, Np)/m # relativistic expression!
            E_K[i] +=  m*sqrt(1 + p^2) * F[i,j] * dp
        end
    end
    return E_K
end

function get_momentum!(u,P,par)
    Nx, dx, Np, dp = par
    F = reshape(u,(Nx,Np+1))
    for i in 1:Nx
        P[i] = 0.0
        for j in 1:Np  
            p = get_p(j, dp, Np) # relativistic expression!
            P[i] +=  p * F[i,j] * dp
        end
    end
    return P
end


function get_E_energy(u,par)
    Nx, dx, Np, dp = par= par
    F = reshape(u,(Nx,Np+1))
    E_E = 0
    for i in 1:Nx
        E_E += F[i,end]^2 * dx
    end
    return E_E/2
end




############################################ TIME INTEGRATION #########################################

function RK4(f,y0,t0,h,p)
    k1 = h*f(y0,t0,p)
    k2 = h*f(y0+0.5*k1, t0+0.5*h,p)
    k3 = h*f(y0+0.5*k2, t0+0.5*h,p)
    k4 = h*f(y0+k3, t0+h,p)
    return y0 + (k1 + 2k2 + 2k3 + k4)./6
end

function RK4_Step!(f,y0,t0,h,p_f,par_RK)
    k1, k2, k3, k4 = par_RK
    k1 = h*f(k1,y0,t0,p_f)
    k2 = h*f(k2,y0+0.5*k1, t0+0.5*h,p_f)
    k3 = h*f(k3,y0+0.5*k2, t0+0.5*h,p_f)
    k4 = h*f(k4,y0+k3, t0+h,p_f)
    y0 .= y0 + (k1 + 2k2 + 2k3 + k4)/6
    return y0
end

function ODEproblem(Method, f, y0, intervalo, M,p)
    ti,tf = intervalo
    h = (tf-ti)/(M-1)
    N = length(y0)
    y = zeros(M,N)
    t = zeros(M)
    y[1,:] = y0
    t[1] = ti
    for i in 2:M
        t[i] = t[i-1] + h
        y[i,:] = Method(f,y[i-1,:],t[i-1],h,p)
    end
    return (t ,y)
end




"""
Plots the values of a matrix as a surface plot.
"""
function plot_matrix(A::Matrix{Float64}; fc=:ocean, linealpha=0.3, fillalpha=0.5, camera=(60,40), title = "")
    default(size=(600,600)
#, fc=:thermal
#, fc=:heat
    , fc=fc
    )
    if !(ndims(A) == 2) 
        error("Array must be 2 dimensional and seems to be of dims = $(ndims(A))")
    end
    (n,m) = size(A)
    x, y = 1:n, 1:m
    z = Surface((x,y)->A[x,y], x, y)
    surface(x,y,z, linealpha = linealpha, fillalpha=fillalpha, display_option=Plots.GR.OPTION_SHADED_MESH, camera=camera, title = title)
end


################################################################ INITIAL DATA #################################


""" Function generate_initial_data!(f, pars_f, u, pars)

    Generates initial data for the distribution function f
        pars_f are the parameters of the function f 
        u is the vector containing the initial data
        pars = (Nx, dx, Np, dp) are the paramters for the initial data.
        
""" 
function generate_initial_data!(f, u, pars_f, pars)
    Nx, dx, Lx, Np, dp, Lp, κ, e = pars 
    F = reshape(u,(Nx,Np+1))
    E = u[Nx*Np+1:Nx*(Np+1)]
    #E = view(F,:,Np+1)
    for i in 1:Nx
        x = get_x(i,dx)
        for j in 1:Np
            p = get_p(j,dp, Np)
            F[i,j] = f(x,p,pars_f)
        end
    end
    ρ = zeros(Nx)
    ϕ = zeros(Nx)
    get_density!(u, ρ, (Nx, dx, Np, dp, m, e))
    n0 = get_total_density!(ρ,(Nx, dx))
    println("n0 = $(n0)")
    global u = u/n0/e # normalize the distribution to value 1 for density.
    get_density!(u, ρ, (Nx, dx, Np, dp, m, e))
    n0 = get_total_density!(ρ,(Nx, dx))
    println("n0 = $(n0)")
    get_ϕ!(ϕ, ρ .- e*n0, κ); # the charge is negative
    u[Nx*Np+1:Nx*(Np+1)] = get_E_from_ϕ!(ϕ, E,dx)
    return u
end

##################################################### EVOLUTION FUNCTION #################################################
"""
Función RHS both, for f and E. 
We include as parameters some vector for holding data and also the velocity vector which is 
given once and for all. v = p/m/sqrt(1+(p/m)^2)
"""
function F!(du,u,t,p_F)
    dx, dp, Nx, Np, v, S, dvx, dvp, D_order = p_F 
    par = (Nx, dx, Np, dp, v, m, e)
    #par_Dx = (Nx, dx, dvx)
    #par_Dp = (Np, dp, dvp)
    par_Dx_ts = (Nx, dx)
    par_Dp_ts = (Np, dp)
    get_current!(u,S,par)
    F = reshape(u,(Nx,Np+1))
    #du .= 0.0 # no es necesario pues toma valores en el primer loop.
    dF = reshape(du,(Nx,Np+1))
    if D_order == 2 
        h_00 = 1/2
        σ = 1/2/h_00/dp
        @threads for j ∈ 1:Np
            @inbounds dF[:,j] = - v[j] * D2x_Per_ts(F[:,j], par_Dx_ts)
        end
        @threads for i ∈ 1:Nx
            @inbounds dF[i,1:Np] += - e * F[i,Np+1] * D2x_SBP_ts(F[i,1:Np], par_Dp_ts) 
            if - e * F[i,Np+1] < 0
                dF[i,1] += - e * F[i,Np+1] * σ * F[i,1]
            else
                dF[i,Np] +=  e * F[i,Np+1] * σ * F[i,Np]
            end
            @inbounds dF[i,Np+1] =  - S[i] 
        end
    elseif D_order == 4
        h_00 = 17/48
        σ = 1/2/h_00/dp
        @threads for j ∈ 1:Np
            @inbounds dF[:,j] = - v[j] * D4x_Per_ts(F[:,j], par_Dx_ts)
        end
        @threads for i ∈ 1:Nx
            @inbounds dF[i,1:Np] += - e * F[i,Np+1] * D4x_SBP_ts(F[i,1:Np], par_Dp_ts,Qd) 
            if - e * F[i,Np+1] < 0
                dF[i,1] += - e * F[i,Np+1] * σ * F[i,1]
            else
                dF[i,Np] +=  e * F[i,Np+1] * σ * F[i,Np]
            end
            @inbounds dF[i,Np+1] =  - S[i] 
        end
    else
        error("Order not defined")
    end
    
    return du[:]
end


#################### DISTRIBUTION FUNCTIONS #################################

"""
function landau_rel_dist(x,p,pars)

    Distribution function for Landau damping. 
    rel_dist(p,θ = θ) * (1. + α *cos(k*x))
    pars = (θ, α, k)
    WARNING THEY ARE NOT NORMALIZED!
"""
function landau_rel_dist(x,p,pars)
    m, θ, α, k = pars 
    return thermal_rel_dist(x, p, (m, θ)) * (1. + α *cos(k*x))
end

"""
function thermal_rel_dist(p, θ = 1, m = 1)
    WARNING THEY ARE NOT NORMALIZED!
"""
function thermal_rel_dist(x, p, (m, θ))
    γ = sqrt(1+p^2/m^2) - 1
    return exp(- γ/θ)/4/π/m
    #falta la función de Bessel!
end

"""
counter_streams_rel_dist(x, p, (m, θ, v))
    we make a double peaked distribution around velocities u = gamma_v(1, +,-v)
"""
function counter_streams_rel_dist(x, p, (m, θ, α, k, v))
    p = p/m
    γ = sqrt(1+p^2)
    γv = 1/sqrt(1-v^2)
    return (exp(- γv*(γ + v*p)/θ) + exp(- γv*(γ - v*p)/θ))*exp(1/θ)/8/π/m* (1. + α *cos(k*x))
end








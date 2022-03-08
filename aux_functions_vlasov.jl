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


################################## FINITE DIFERENCE OPERATORS #######

function D2x_Per(v,par_Dx)
    N, dx, dv = par_Dx
    for i in 1:N
        dv[i] = (v[mod1((i+1), N)] - v[mod1((i + N -1), N)])/dx/2
    end
    return dv[:]
end

function D2x_SBP(v,par_Dx)
    N, dx, dv = par_Dx
    dv[1] = (v[2] - v[1])/dx
    for j in 2:(N-1)
        dv[j] = (v[j+1] - v[j-1])/dx/2
    end
    dv[N] = (v[N] - v[N-1])/dx
    return dv[:]
end

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
    V[j] = 4 * π * V[j] / (j-1)^2 / κ^2
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
    Nx, dx, Np, dp, m, e = par
    F = reshape(u,(Nx,Np+1))
    for i in 1:Nx
        S[i] = 0
        for j in 1:Np
            p = get_p(j, dp, Np)/m
            S[i] += e * F[i,j]* p/sqrt(1+p^2) * dp
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
    Nx, dx = par
    F = reshape(u,(Nx,Np+1))
    E_E = 0
    for i in 1:Nx
        E_E += F[i,end]^2 * dx
    end
    return E_E/8/π
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
function plot_matrix(A,fc = :ocean, linealpha = 0.3, fillalpha = 0.5, camera = (60,40))
    default(size=(600,600)
#, fc=:thermal
#, fc=:heat
    , fc=:ocean
    )
    if !(ndims(A) == 2) 
        error("Array must be 2 dimensional and seems to be of dims = $(ndims(A))")
    end
    (n,m) = size(A)
    x, y = 1:n, 1:m
    z = Surface((x,y)->A[x,y], x, y)
    surface(x,y,z, linealpha = 0.3, fillalpha=0.5, display_option=Plots.GR.OPTION_SHADED_MESH, camera=(60,40))
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
"""
function F!(du,u,t,p_F)
    dx, dp, Nx, Np, S, dvx, dvp = p_F 
    par = (Nx, dx, Np, dp, m, e)
    par_Dx = (Nx, dx, dvx)
    par_Dp = (Np, dp, dvp)
    get_current!(u,S,par)
    F = reshape(u,(Nx,Np+1))
    dF = reshape(du,(Nx,Np+1))
    du .= 0.0
    for j ∈ 1:Np
        p = get_p(j, dp, Np)/m
        dF[:,j] = - p/sqrt(1+ p^2) * D2x_Per(F[:,j], par_Dx)
    end
    for i ∈ 1:Nx
        dF[i,1:Np] +=  e * F[i,Np+1] * D2x_SBP(F[i,1:Np], par_Dp) 
        dF[i,Np+1] =  4* 4π * S[i] 
    end
    return du
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
    θ, m, α, k = pars 
    return thermal_rel_dist(x, p, (m, θ)) * (1. + α *cos(k*x))
end

"""
function thermal_rel_dist(p, θ = 1, m = 1)
    WARNING THEY ARE NOT NORMALIZED!
"""
function thermal_rel_dist(x, p, (m, θ))
    γ = sqrt(1+p^2/m^2)
    return exp(- γ/θ)/4/π/m^3
    #falta la función de Bessel!
end








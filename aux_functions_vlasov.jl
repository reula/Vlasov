# Derivadas
get_x(i,dx) = (i-1)*dx 
get_p(j,dp) = (j-1)*dp 

function D2x_Per(v,par_Dx)
    dx, N, dv = par_Dx
    for i in 1:N
        dv[i,:] = (v[mod1((i+1), N),:] - v[mod1((i + N -1), N),:])/dx/2
    end
    return dv[:,:]
end

function D2p_SBP(v,par_Dp)
    dx, N, dv = par_Dp
    dv[:,1] = (v[:,2,:] - v[1,:])/dx
    for j in 2:(N-1)
        dv[j,:] = (v[:,j+1,:] - v[j-1,:])/dx/2
    end
    dv[:,N] = (v[:,N,:] - v[N-1,:])/dx
    return dv[:,:]
end


"""
The following function evaluates the electric field on a uniform grid from the electric potential.

    // Calculate electric field from potential
"""
function get_E_from_ϕ!(ϕ, E, dx)
      J = length(E)
      for j in 2:J-1
        E[j] = (ϕ[j-1] - ϕ[j+1]) / 2. / dx
      end
      E[1] = (ϕ[J] - ϕ[2]) / 2. / dx;
      E[J] = (ϕ[J-1] - ϕ[1]) / 2. / dx;
end

""" The following routine solves Poisson's equation in 1-D to find the instantaneous electric potential on a uniform grid.

// Solves 1-d Poisson equation:
//    d^u / dx^2 = v   for  0 <= x <= L
// Periodic boundary conditions:
//    u(x + L) = u(x),  v(x + L) = v(x)
// Arrays u and v assumed to be of length J.
// Now, jth grid point corresponds to
//    x_j = j dx  for j = 0,J-1
// where dx = L / J. L / (J-1) in Julia
// Also,
//    kappa = 2 pi / L
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
    V[j] = - V[j] / (j-1)^2 / κ^2
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

# CURRENT AND DENSITY

function get_current!(u,S,par)
    Nx, dx, Np, dp, m = par
    for i in 1:Nx
        S[i] = 0
        for j in 2:(Np-1)  
            p = (j-1)*dp/m
            S[i] = S[i] + u[i,j]*p*dp/sqrt(1+p^2)
        end
        p = (Np-1)*dp/m
        S[i] = S[i] + u[i,Np]*p*dp/(1+p^2)/2
    end
    return S
end

function get_density!(u,ρ,par)
    Nx, dx, Np, dp, m = par
    for i in 1:Nx
        ρ[i] = 0
        for j in 2:(Np-1)  
            ρ[i] = ρ[i] + u[i,j]*dp
        end
        ρ[i] = ρ[i] + (u[i,1] + u[i,Np])/2*dp
    end
    return ρ
end

function get_total_density!(ρ,par)
    Nx, dx = par
    n0 = 0.0
    for i in 1:Nx
        n0 += ρ[i] 
    end
    return n0/Nx 
end

function get_K_energy!(u,E_K,par)
    Nx, dx, Np, dp = par
    for i in 1:Nx
        E_K[i] = 0.0
        for j in 2:(Np-1)  
            p = (j-1) * dp / m # relativistic expression!
            E_K[i] +=  m*(sqrt(1 + p^2) - 1) * u[i,j] * dp
        end
        p = (Np-1)*dp
        E_K[i] += m*(sqrt(1 + p^2) - 1) * u[i,Np] / 2 * dp
    end
    return E_K
end

function get_momentum!(u,P,par)
    Nx, dx, Np, dp = par
    for i in 1:Nx
        P[i] = 0.0
        for j in 2:(Np-1)  
            p = (j-1) * dp # relativistic expression!
            P[i] +=  p * u[i,j] * dp
        end
        p = (Np-1)*dp
        P[i] += p * u[i,Np] / 2 * dp
    end
    return P
end


function get_E_energy!(u,E_E,par)
    Nx, dx = par
    E_E = 0
    for i in 1:Nx
        E_E += u[i,end]^2 * dx
    end
    return E_E
end




# TIME INTEGRATION

function RK4(f,y0,t0,h,p)
    k1 = h*f(y0,t0,p)
    k2 = h*f(y0+0.5*k1, t0+0.5*h,p)
    k3 = h*f(y0+0.5*k2, t0+0.5*h,p)
    k4 = h*f(y0+k3, t0+h,p)
    return y0 + (k1 + 2k2 + 2k3 + k4)./6
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


function rel_dist(p;m=1,θ=1)
    γ = sqrt(1+p^2/m^2)
    return exp(- γ/θ)/4/π/m^3
    #falta la función de Bessel!
end


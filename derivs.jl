################################## FINITE DIFERENCE OPERATORS #######

function D2x_Per(v,par_Dx)
    N, dx, dv = par_Dx
    dv[1] = (v[2] - v[N])/dx/2
    @inbounds for i in 2:(N-1)
        dv[i] = (v[i+1] - v[i-1])/dx/2
    end
    dv[N] = (v[1] - v[N-1])/dx/2
    return dv[:]
end

function D4x_Per(v,par_Dx)
    N, dx, dv = par_Dx
    dv[1] = (-v[3] + 8.0 * v[2] - 8.0 * v[N] + v[N-1])/dx/12
    dv[2] = (-v[4] + 8.0 * v[3] - 8.0 * v[1] + v[N])/dx/12
    @inbounds for i in 3:(N-2)
        dv[i] = (-v[i+2] + 8.0*v[i+1] - 8.0 * v[i-1] + v[i-2] )/dx/12
    end
    dv[N] = (-v[2] + 8.0 * v[1] - 8.0 * v[N-1] + v[N-2])/dx/12
    dv[N-1] = (-v[1] + 8.0 * v[N] - 8.0 * v[N-2] + v[N-3])/dx/12
    return dv[:]
end

function D2x_Per_ts(v,par_Dx)
    N, dx = par_Dx
    dv = Vector{Float64}(undef,N)
    dv[1] = (v[2] - v[N])/dx/2
    @inbounds for i in 2:(N-1)
        dv[i] = (v[i+1] - v[i-1])/dx/2
    end
    dv[N] = (v[1] - v[N-1])/dx/2
    return dv[:]
end

function D4x_Per_ts(v,par_Dx)
    N, dx = par_Dx
    dv = Vector{Float64}(undef,N)
    dv[1] = (-v[3] + 8.0 * v[2] - 8.0 * v[N] + v[N-1])/dx/12
    dv[2] = (-v[4] + 8.0 * v[3] - 8.0 * v[1] + v[N])/dx/12
    @inbounds for i in 3:(N-2)
        dv[i] = (-v[i+2] + 8.0*v[i+1] - 8.0 * v[i-1] + v[i-2] )/dx/12
    end
    dv[N] = (-v[2] + 8.0 * v[1] - 8.0 * v[N-1] + v[N-2])/dx/12
    dv[N-1] = (-v[1] + 8.0 * v[N] - 8.0 * v[N-2] + v[N-3])/dx/12
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

function D2x_SBP_ts(v,par_Dx_st)
    N, dx = par_Dx_st
    dv = Vector{Float64}(undef,N)
    dv[1] = (v[2] - v[1])/dx
    for j in 2:(N-1)
        dv[j] = (v[j+1] - v[j-1])/dx/2
    end
    dv[N] = (v[N] - v[N-1])/dx
    return dv[:]
end

#### fourth order derivative coefficients ###############################

const Qd = [-24/17 59/34 -4/17 -3/34    0  0 
            -1/2   0    1/2    0      0  0
            8/86  -59/86  0   59/86  -8/86 0
            3/98   0   -59/98  0 32/49  -4/49]

function D4x_SBP(v,par_Dx,Qd)
    N, dx, dv = par_Dx
    dv[1] = (v[1]*Qd[1,1] + v[2]*Qd[1,2] + v[3]*Qd[1,3] + v[4]*Qd[1,4])/dx
    dv[2] = (v[1]*Qd[2,1] + v[3]*Qd[2,3])/dx
    dv[3] = (v[1]*Qd[3,1] + v[2]*Qd[3,2] + v[4]*Qd[3,4] + v[5]*Qd[3,5])/dx
    dv[4] = (v[1]*Qd[4,1] + v[3]*Qd[4,3] + v[5]*Qd[4,5] + v[6]*Qd[4,6])/dx
    for i in 5:N-4
        dv[i] = (-v[i+2] + 8.0*v[i+1] - 8.0 * v[i-1] + v[i-2] )/dx/12
    end
    dv[N]   = -(v[N]*Qd[1,1] + v[N-1]*Qd[1,2] + v[N-2]*Qd[1,3] + v[N-3]*Qd[1,4])/dx
    dv[N-1] = -(v[N]*Qd[2,1] + v[N-2]*Qd[2,3])/dx
    dv[N-2] = -(v[N]*Qd[3,1] + v[N-1]*Qd[3,2] + v[N-3]*Qd[3,4] + v[N-4]*Qd[3,5])/dx
    dv[N-3] = -(v[N]*Qd[4,1] + v[N-2]*Qd[4,3] + v[N-4]*Qd[4,5] + v[N-5]*Qd[4,6])/dx
    return dv[:]
end

function D4x_SBP_ts(v,par_Dx,Qd)
    N, dx = par_Dx
    dv = Vector{Float64}(undef,N)
    dv[1] = (v[1]*Qd[1,1] + v[2]*Qd[1,2] + v[3]*Qd[1,3] + v[4]*Qd[1,4])/dx
    dv[2] = (v[1]*Qd[2,1] + v[3]*Qd[2,3])/dx
    dv[3] = (v[1]*Qd[3,1] + v[2]*Qd[3,2] + v[4]*Qd[3,4] + v[5]*Qd[3,5])/dx
    dv[4] = (v[1]*Qd[4,1] + v[3]*Qd[4,3] + v[5]*Qd[4,5] + v[6]*Qd[4,6])/dx
    for i in 5:N-4
        dv[i] = (-v[i+2] + 8.0*v[i+1] - 8.0 * v[i-1] + v[i-2] )/dx/12
    end
    dv[N]   = -(v[N]*Qd[1,1] + v[N-1]*Qd[1,2] + v[N-2]*Qd[1,3] + v[N-3]*Qd[1,4])/dx
    dv[N-1] = -(v[N]*Qd[2,1] + v[N-2]*Qd[2,3])/dx
    dv[N-2] = -(v[N]*Qd[3,1] + v[N-1]*Qd[3,2] + v[N-3]*Qd[3,4] + v[N-4]*Qd[3,5])/dx
    dv[N-3] = -(v[N]*Qd[4,1] + v[N-2]*Qd[4,3] + v[N-4]*Qd[4,5] + v[N-5]*Qd[4,6])/dx
    return dv[:]
end
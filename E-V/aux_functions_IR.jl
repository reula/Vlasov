"""
Evolution function for the scalar wave system.
    The parameters are:
    D SummationByPartsOperators
    Δ Dissipation Operator (ussually associated to D)
    x array of points where derivatives are computed
    dx 
    boundary function is A*sin(ωt)*exp(t/τ)
    σ is coefficient in front of Δ
"""
function F!(du,u,p,t)
    # second order version
    D, Δ, x, dx, A, ω, τ, σ = p
    
    f = @view u[1:N]
    g = @view u[N+1:2N]
    h = @view u[2N+1:3N]
    χ₊ = @view u[3N+1:4N]
    χ₋ = @view u[4N+1:5N]

    df = @view du[1:N]
    dg = @view du[N+1:2N]
    dh = @view du[2N+1:3N]
    dχ₊ = @view du[3N+1:4N]
    dχ₋ = @view du[4N+1:5N]

    #= this is only the original system, (4) not the one we actually evolve!
    @. df = D*f 
    @. dg = D*g
    @. dh = D*h - df/2/f/x + (g-2)/2/f/h/x * dg - (g-f)*h/f/x -(g-2+f)/2/f/x^2 + 2*((χ₊ + χ₋)/f/x)^2 
    @. dg = -g/f * df + (f-2)/f * dg + 2g*h -g*(g-2+f)/f/x  
    @. df = df + (g-2+f)/x 
    @. dχ₊ = D*χ₊ - (f-2)* χ₋ /f/x
    @. dχ₋ = (f-2)/f * D*χ₋ 2χ₋ *(2x*(f-2)*h + 2 - g -f)/f^2/x + 8χ₋*(χ₊)^2/f^3/x - χ₊/x
    =#

    # these are the equations, (7)
    @. df = 2*(f-2)*h + 4* (χ₊)^2 /f/x
    @. dg = 2*g*h - 2g* (χ₊)^2 /f/x + 2g*(f-2)* (χ₋)^2/ f^2/x 
    
    #dh = D * h
    mul!(dh,D,h)
    #dh = D * h - σ*Δ * h
    @. dh = dh + (- (g-2)* (χ₊)^2 + 4χ₊*χ₋ + g* (χ₋)^2 ) /f^2/x^2
    dh[end] = dh[end] - 1.0/right_boundary_weight(D) *h[end] #penalty BC
    mul!(dh,Δ,h,-σ,true)

    #dχ₊ = D * χ₊
    mul!(dχ₊,D,χ₊) 
    #dχ₊ = D * χ₊ - σ*Δ * χ₊
    @. dχ₊ += - (f-2)* χ₋ /f/x
    dχ₊[end] = dχ₊[end] - 1.0/right_boundary_weight(D) * (χ₊[end] - A*sin(ω*t)*exp(-t/τ)) #penalty BC
    mul!(dχ₊,Δ,χ₊,-σ,true)

    #dχ₋ = D * χ₋
    mul!(dχ₋,D, χ₋)
    @. dχ₋ = (f-2)/f * dχ₋ + 2χ₋ *(2x*(f-2)*h + 2 - g -f)/f^2/x + 8χ₋*(χ₊)^2/f^3/x - χ₊/x
    
    if f[1]<2.0
        #dχ₋[1] += (f[1]-2)/f[1]/left_boundary_weight(D) *(χ₋[1] + χ₊[1]) #penalty BC
        dχ₋[1] += (f[1]-2)/f[1]/left_boundary_weight(D) * χ₋[1] #penalty BC
    end
    mul!(dχ₋,Δ,χ₋,-σ,true)

    #@. du = [df; dg; dh; dχ₊; dχ₋]
end

"""
Computes the constraint equations for that give a derivative operator and the fields. 
    Returns values in the vector C=[Cf;Cg]
    So previously create a vector of size 2N
    """
function constraints!(C,u,D)
    x = SummationByPartsOperators.grid(D)
    N = length(x)
    f = @view u[1:N]
    g = @view u[N+1:2N]
    h = @view u[2N+1:3N]
    χ₊ = @view u[3N+1:4N]
    χ₋ = @view u[4N+1:5N]
    Cf = @view C[1:N]
    Cg = @view C[N+1:2N]
    #Cf = D * f
    mul!(Cf,D,f)
    #Cg = D * g
    mul!(Cg,D,g)
    @. Cf += - 2h*(f-2) + (g + f -2)/x - 4*(χ₊)^2/f/x 
    @. Cg += - 2g*(h*f*x - (χ₊)^2 + (χ₋)^2)/f/x
    #C = [Cf;Cg]
end

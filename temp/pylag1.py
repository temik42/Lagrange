import numpy as np

def d(x, order = 1):
    if (order == 1):
        return np.gradient(x, edge_order = 2)
    if (order == 2):
        out = np.roll(x,1)+np.roll(x,-1)-2*x
        out[0] = 2*x[0] - 5*x[1] + 4*x[2] - 1*x[3]
        out[-1] = 2*x[-1] - 5*x[-2] + 4*x[-3] - 1*x[-4]
        return out


def f(t,x,v,T):
    f1 = d(v)
    f1[[0,-1]] = 0
    f2 = 2*d(x)/(x)**2*T-2*d(T)/x
    f2[[0,1,-1,-2]] = 0
    f3 = -2./3*T*d(v)/x + (d(T,2)*x-d(x)*d(T))/(x)**3
    f3[[0,1,-1,-2]] = 0
    return f1,f2,f3

def step(dt,t,x,v,T,f):
    x0 = np.array(x)
    
    f0 = f(t,x,v,T)
    x1 = x + 0.5*dt*f0[0]
    v1 = v + 0.5*dt*f0[1]
    T1 = T + 0.5*dt*f0[2]
    f1 = f(t+0.5*dt,x1,v1,T)
    x2 = x + 0.5*dt*f1[0]
    v2 = v + 0.5*dt*f1[1]
    T2 = T + 0.5*dt*f1[2]
    f2 = f(t+0.5*dt,x2,v2,T)
    x3 = x + dt*f2[0]
    v3 = v + dt*f2[1]
    T3 = T + dt*f2[2]
    f3 = f(t+dt,x3,v3,T)
    
    x += dt/6*(f0[0] + 2*f1[0] + 2*f2[0] + f3[0])
    v += dt/6*(f0[1] + 2*f1[1] + 2*f2[1] + f3[1])
    T += dt/6*(f0[2] + 2*f1[2] + 2*f2[2] + f3[2])
    
    v[[0,-1]] = 0
    #v[[0,-1]] = -v[[0,-1]]
    T[[0,-1]] = 1.
    t += dt
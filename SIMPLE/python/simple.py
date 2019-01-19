##Channel Flow Navier Stokes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def build_up_b(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    
    return b


def pressure_poisson_periodic(p, dx, dy, pts):
    pn = np.empty_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        # Wall/object boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        for pair in pts:
            i = pair[0]
            j = pair[1]
            p[i,j] = p[i-1, j]
    
    return p


def set_obj_bounds(u,v,pts):
    for pair in pts:
        i = pair[0]
        j = pair[1]
        u[i,j] = 0
        v[i,j] = 0


if __name__ == '__main__':
    xmin = 0
    xmax = 2.0
    ymin = 0
    ymax = 2.0
    epsilon = 0.00001

    nx = 51
    ny = 51
    nt = 10
    nit = 50

    dx = xmax/(nx-1)
    dy = ymax/(ny-1)
    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)
    Y,X = np.meshgrid(y,x)

    print(ny, dy)
    # physical variables -- Re = (rho*D*F)/nu
    rho = 1     # fluid density
    nu = .1     # fluid viscosity
    F = 1       # x-velocity
    dt = .01    # timestep


    #initial conditions
    u = np.zeros((ny,nx)) ##create a XxY vector of 0's
    un = np.zeros((ny,nx)) ##create a XxY vector of 0's

    v = np.zeros((ny,nx)) ##create a XxY vector of 0's
    vn = np.zeros((ny,nx)) ##create a XxY vector of 0's

    p = np.ones((ny,nx)) ##create a XxY vector of 0's
    pn = np.ones((ny,nx)) ##create a XxY vector of 0's

    b = np.zeros((ny,nx))
    
    r = 0.25
    xc = 1
    yc = 1
    objPts = []
    for i in range(0, u.shape[0]):
        for j in range(0, u.shape[1]):
            x = dx*i
            y = dy*j
            dsq = (x - xc)**2 + (y - yc)**2
            if dsq <= (r**2):
                objPts.append([i, j])
                
#    for pair in objPts:
#        i = pair[0]
#        j = pair[1]
#        b[i,j] = 1
    
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(threshold=np.nan)
#    print(b)
#    exit()
    
    udiff = 1
    stepcount = 0

    while udiff > epsilon:
        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic(p, dx, dy, objPts)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx * 
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy * 
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                        dt / (2 * rho * dx) * 
                        (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                        nu * (dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        dt / dy**2 * 
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                        F * dt)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx * 
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy * 
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * 
                        (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 * 
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Periodic BC u @ x = 2     
        u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
                    (un[1:-1, -1] - un[1:-1, -2]) -
                    vn[1:-1, -1] * dt / dy * 
                    (un[1:-1, -1] - un[0:-2, -1]) -
                    dt / (2 * rho * dx) *
                    (p[1:-1, 0] - p[1:-1, -2]) + 
                    nu * (dt / dx**2 * 
                    (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                    dt / dy**2 * 
                    (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

        # Periodic BC u @ x = 0
        u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                    (un[1:-1, 0] - un[1:-1, -1]) -
                    vn[1:-1, 0] * dt / dy * 
                    (un[1:-1, 0] - un[0:-2, 0]) - 
                    dt / (2 * rho * dx) * 
                    (p[1:-1, 1] - p[1:-1, -1]) + 
                    nu * (dt / dx**2 * 
                    (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                    dt / dy**2 *
                    (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

        # Periodic BC v @ x = 2
        v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                    (vn[1:-1, -1] - vn[1:-1, -2]) - 
                    vn[1:-1, -1] * dt / dy *
                    (vn[1:-1, -1] - vn[0:-2, -1]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, -1] - p[0:-2, -1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                    dt / dy**2 *
                    (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                    (vn[1:-1, 0] - vn[1:-1, -1]) -
                    vn[1:-1, 0] * dt / dy *
                    (vn[1:-1, 0] - vn[0:-2, 0]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, 0] - p[0:-2, 0]) +
                    nu * (dt / dx**2 * 
                    (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                    dt / dy**2 * 
                    (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))


        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0
        #print(u)
        set_obj_bounds(u,v,objPts)
        
        
        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        stepcount += 1
    
    print(stepcount)
    plt.quiver(X, Y, u, v)
    plt.show()

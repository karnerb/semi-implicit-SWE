import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams




n=513
m=int((n+1)/2+1)

#parametri iz naloge
g=10
#višina gladine
H0=10000
#višina perturbacije
Z0=100
#dolžina domene
L=np.cos(50/90*np.pi/2)*6400*1000*2*np.pi*1/3
#delta x = delta y
D=L/(n-1)
#časovni korak
dt=2*D/np.sqrt(H0*g)

#mreža
x = np.linspace(-(n + 1) / 2 * D, (n + 1) / 2 * D, n + 4)
y = np.linspace(-(m + 1) / 2 * D, (m + 1) / 2 * D, m + 4)
xx, yy = np.meshgrid(y, x)

#coriolisov parameter
f0 = 2*(2*np.pi)/(24*60*60)*np.sin(50/90*np.pi/2)
beta = 2*(2*np.pi)/(24*60*60)*np.cos(50/90*np.pi/2)/6400000

fcol = f0 + beta*y
f=np.zeros((n+4,m+4))
for i in range(n+4):
    f[i, :] = fcol

days = 2
tend = days*24*3600
#set output every nsteps
nsteps=1
#add nonlinear convection terms
nonlinear = False
upwind=False
#linear friction coefficient
eps=0

#initial gaussian bump
def gauss(xx,yy):
    return Z0*np.exp(-(xx*xx+yy*yy) / (2.0*(0.05*L)**2))
#eigenmode of linearized SWE
def sine(xx, yy):
    return Z0*np.sin(2*np.pi/((n-1)*D)*yy)*np.sin(2*np.pi/((m-1)*D)*xx)

#0 for periodic bc, 1 for periodic in x, dh/dy = 0, u=0, v=0 in y
bc = 1
def apply_bc(Z, U, V):
    #periodic in x
    if bc==0:
        Z[0, :] = Z[n, :]
        Z[1, :] = Z[n + 1, :]
        Z[n + 2, :] = Z[2, :]
        Z[n + 3, :] = Z[3, :]
        U[0, :] = U[n, :]
        U[1, :] = U[n + 1, :]
        U[n + 2, :] = U[2, :]
        U[n + 3, :] = U[3, :]
        V[0, :] = V[n, :]
        V[1, :] = V[n + 1, :]
        V[n + 2, :] = V[2, :]
        V[n + 3, :] = V[3, :]

        Z[:, 0] = Z[:, m]
        Z[:, 1] = Z[:, m + 1]
        Z[:, m + 2] = Z[:, 2]
        Z[:, m + 3] = Z[:, 3]
        U[:, 0] = U[:, m]
        U[:, 1] = U[:, m + 1]
        U[:, m + 2] = U[:, 2]
        U[:, m + 3] = U[:, 3]
        V[:, 0] = V[:, m]
        V[:, 1] = V[:, m + 1]
        V[:, m + 2] = V[:, 2]
        V[:, m + 3] = V[:, 3]

    if bc == 1:
        Z[0, :] = Z[n, :]
        Z[1, :] = Z[n + 1, :]
        Z[n + 2, :] = Z[2, :]
        Z[n + 3, :] = Z[3, :]
        U[0, :] = U[n, :]
        U[1, :] = U[n + 1, :]
        U[n + 2, :] = U[2, :]
        U[n + 3, :] = U[3, :]
        V[0, :] = V[n, :]
        V[1, :] = V[n + 1, :]
        V[n + 2, :] = V[2, :]
        V[n + 3, :] = V[3, :]

        Z[:, 0] = Z[:, 3]
        Z[:, 1] = Z[:, 2]
        Z[:, m + 2] = Z[:, m+1]
        Z[:, m + 3] = Z[:, m]
        U[:, 0] = 0
        U[:, 1] = 0
        U[:, m + 2] = 0
        U[:, m + 3] = 0
        V[:, 0] = 0
        V[:, 1] = 0
        V[:, m + 2] = 0
        V[:, m + 3] = 0

    return Z, U, V

#2n order central differences
def dx(f):
    return 1/(2*D) * (np.roll(f,-1,axis=0)-np.roll(f,1,axis=0))
def dy(f):
    return 1 / (2 * D) * (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))
def nabla2(f):
    return dx(dx(f))+dy(dy(f))
def dxy(f):
    return dx(dy(f))
def dxx(f):
    return 1/(4*D*D) * (np.roll(f,-2,axis=0)+np.roll(f,2,axis=0)-2*f)
def dyy(f):
    return 1/(4*D*D) * (np.roll(f,-2,axis=1)+np.roll(f,2,axis=1)-2*f)
def dxx1(f):
    return 1/(4*D*D)*(np.roll(f, -2, axis=0) + np.roll(f, 2, axis=0))
def dyy1(f):
    return 1/(4*D*D)*(np.roll(f, -2, axis=1)+np.roll(f, 2, axis=1))

#upwind differences for advection term
def upwind_dx_negative(f):
    return 1/D * (np.roll(f, -1, axis=0)-f)
def upwind_dx_positive(f):
    return 1 / D * (f-np.roll(f, 1, axis=0))
def upwind_dy_negative(f):
    return 1/D * (np.roll(f, -1, axis=1)-f)
def upwind_dy_positive(f):
    return 1 / D * (f-np.roll(f, 1, axis=1))

#calculate energy, mass, enstrophy
def calc_emp(zn, un, vn):
    zeta = H0 + zn[2:n + 2, 2:m + 2]
    E = 0.5 * g * np.sum(zeta * zeta) + 0.5 * np.sum(
        zeta * (
                un[2:n + 2, 2:m + 2] * un[2:n + 2, 2:m + 2] + vn[2:n + 2, 2:m + 2] * vn[2:n + 2, 2:m + 2]))
    M = np.sum(zeta)

    P = np.sum(1 / (2 * zeta) * np.square(f[2:n + 2, 2:m + 2] + 1 / (2 * D) * (
                np.roll(vn, 1, axis=0)[2:n + 2, 2:m + 2] - np.roll(vn, -1, axis=0)[2:n + 2, 2:m + 2] - np.roll(un, 1,
                                                                                                               axis=1)[
                                                                                                       2:n + 2,
                                                                                                       2:m + 2] + np.roll(
            un, -1, axis=1)[2:n + 2, 2:m + 2])))

    return E,M,P

#solves helmholtz equation in zn for even time steps
def iterate_even(Zn, Un, Vn, Az, Au, Av):
    delta = 10
    zold=Zn.copy()
    while (delta>0.0001):
        zprev = Zn.copy()
        Zn = (zold - H0 * dt / 2 * (dx(Un) +  dt * dx(f * Vn) - g * dt / 2 * (dxx1(Zn) + dxx(zold)) + dt * dx(Au)
                                    + dy(Vn) -  dt * (dy(f * Un) +  dt * dy(f*f*Vn) - g * dt / 2 * (dy(f*dx(Zn)) + dy(f*dx(zold))) + dy(f*Au) * dt)
                                    - g * dt / 2 * (dyy1(Zn) + dyy(zold)) + dy(Av) * dt + dx(Un) + dy(Vn))
              + Az * dt
              ) / (1+g*H0*dt*dt/(4*D*D))
        Zn, trash1, trash2 = apply_bc(Zn, np.zeros((n + 4, m + 4)), np.zeros((n + 4, m + 4)))
        delta = np.sum(np.abs(zprev - Zn))
    return Zn
#solves helmholtz equation in zn for odd time steps
def iterate_odd(Zn, Un, Vn, Az, Au, Av):
    delta = 10
    zold=Zn.copy()
    while (delta>0.0001):
        zprev = Zn.copy()
        Zn = (zold - H0 * dt / 2 * (dx(Un) +  dt * (dx(f * Vn) -  dt * dx(f * f * Un) - g * dt / 2 * (dx(f*dy(Zn)) + dx(f*dy(zold))) + dt * dx(f*Av)) - g * dt / 2 * (dxx1(Zn) + dxx(zold)) + dx(Au) * dt
                                    + dy(Vn) -  dt * dy(f * Un) - g * dt / 2 * (dyy1(Zn) + dyy(zold)) + dy(Av) * dt
                                    + dx(Un) + dy(Vn))
              + Az * dt
              ) / (1+g*H0*dt*dt/(4*D*D))

        Zn, trash1, trash2 = apply_bc(Zn, np.zeros((n + 4, m + 4)), np.zeros((n + 4, m + 4)))
        delta = np.sum(np.abs(zprev - Zn))
    return Zn

#integration and plotting
def semi_implicit(zn, un, vn):

    E0, M0, P0 = calc_emp(zn,un,vn)

    # zn = gauss(xx,yy)
    # un = np.zeros((n + 4, m + 4))
    # vn = np.zeros((n + 4, m + 4))

    zn, un, vn = apply_bc(zn, un, vn)

    znp1=np.zeros((n+4, m+4))
    unp1=np.zeros((n+4, m+4))
    vnp1 = np.zeros((n + 4, m + 4))

    t=0
    i=0
    while (t<tend):
        print('t={}'.format(t/60/60))
        #plotting
        if (i % nsteps == 0):
            #conserved quantities - energy, mass, enstrophy
            E, M, P = calc_emp(zn, un, vn)

            #plot z
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_box_aspect([1,0.5,1])
            ax.set_zlim(-1.1 * Z0, 1.1 * Z0)

            # ax.set_axis_off()

            plt.title('t={0:.3f}'.format(t))
            plt.title('$t={0:.2f}h$, $M={1:.4f}$, $E={2:.4f}$,'
                      '\n$P={3:.4f}$'.format(t/(60*60), M/M0, E/E0, P/P0))
            plt.xlabel('x')
            plt.ylabel('y')
            surf = ax.plot_surface(yy[2:n + 2, 2:m + 2], xx[2:n + 2, 2:m + 2], zn[2:n + 2, 2:m + 2], cmap=cm.coolwarm,
                                   linewidth=0, antialiased=True)

            plt.savefig('animacija2/fig{}.png'.format(str(i).zfill(4)))
            plt.close()

            #plot velocities
            plt.quiver(yy[2:n + 2, 2:m + 2][::8, ::8], xx[2:n + 2, 2:m + 2][::8, ::8], un[2:n + 2, 2:m + 2][::8, ::8],vn[2:n + 2, 2:m + 2][::8, ::8], scale=20)
            plt.title('$t={0:.2f}h$, $M={1:.4f}$, $E={2:.4f}$,'
                      '\n$P={3:.4f}$'.format(t/(60*60), M/M0, E/E0, P/P0))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig('animacija3/fig{}.png'.format(str(i).zfill(4)))

            plt.close()

            # #plot height contours
            # plt.contourf(yy[2:n + 2, 2:m + 2], xx[2:n + 2, 2:m + 2], zn[2:n + 2, 2:m + 2])
            # plt.savefig('animacija4/fig{}.png'.format(str(i).zfill(4)))
            # plt.close()

            # plot curl(velocitiy)
            plt.matshow(np.log(np.abs(dx(D*vn)-dy(D*un))+0.00001)[3:n + 1, 3:m + 1].T)
            plt.axis('off')
            plt.colorbar()

            plt.title('Logaritem absolutne vrtinčnosti\n' + '$t={0:.2f}h$, $M={1:.4f}$, $E={2:.4f}$,'
                      '\n$P={3:.4f}$'.format(t / (60 * 60), M / M0, E / E0, P / P0))

            plt.savefig('animacija5/fig{}.png'.format(str(i).zfill(4)))
            plt.close()

        #if !nonlinear
        Az=np.zeros((n+4,m+4))
        Au = np.zeros((n + 4, m + 4))
        Av = np.zeros((n + 4, m + 4))

        if nonlinear:

            if upwind:
                Az = -un*upwind_dx_positive(zn)*np.ma.masked_greater_equal(un, 0).mask - un*upwind_dx_negative(zn)*np.ma.masked_less_equal(un, 0).mask\
                    + -vn*upwind_dy_positive(zn)*np.ma.masked_greater_equal(vn, 0).mask - vn*upwind_dy_negative(zn)*np.ma.masked_less_equal(vn, 0).mask\
                    + -zn*upwind_dx_positive(un)*np.ma.masked_greater_equal(zn, 0).mask - zn*upwind_dx_negative(un)*np.ma.masked_less_equal(zn, 0).mask\
                    + -zn*upwind_dy_positive(vn)*np.ma.masked_greater_equal(zn, 0).mask - zn*upwind_dy_negative(vn)*np.ma.masked_less_equal(zn, 0).mask\
                    - eps*(H0+zn)


                Au = -un*upwind_dx_positive(un)*np.ma.masked_greater_equal(un, 0).mask - un*upwind_dx_negative(un)*np.ma.masked_less_equal(un, 0).mask\
                    -vn*upwind_dy_positive(un)*np.ma.masked_greater_equal(vn, 0).mask - vn*upwind_dy_negative(un)*np.ma.masked_less_equal(vn, 0).mask\
                    -eps*un

                Av = -un*upwind_dx_positive(vn)*np.ma.masked_greater_equal(un, 0).mask - un*upwind_dx_negative(vn)*np.ma.masked_less_equal(un, 0).mask\
                     -vn*upwind_dy_positive(vn)*np.ma.masked_greater_equal(vn, 0).mask - vn*upwind_dy_negative(vn)*np.ma.masked_less_equal(vn, 0).mask\
                     -eps*vn

            else:
                Az = -dx(zn*un)-dy(zn*vn)-eps*(H0+zn)
                Au = -(un*dx(un)+vn*dy(un))-eps*un
                Av= -(un*dx(vn)+vn*dy(vn))-eps*vn




        #odd/even timesteps
        if (i%2==0):
            znp1 = iterate_even(zn, un, vn, Az, Au, Av)
            unp1 = un + f*dt*vn-g*dt/2*(dx(znp1)+dx(zn))+Au*dt
            vnp1 = vn - f*dt*unp1-g*dt/2*(dy(znp1)+dy(zn))+Av*dt
        if (i%2 == 1):
            znp1 = iterate_odd(zn, un, vn, Az, Au, Av)
            vnp1 = vn - f * dt * un - g * dt / 2 * (dy(znp1) + dy(zn)) + Av * dt
            unp1 = un +f*dt*vnp1-g*dt/2*(dx(znp1)+dx(zn))+Au*dt


        zn=znp1.copy()
        un=unp1.copy()
        vn=vnp1.copy()

        zn, un, vn = apply_bc(zn, un, vn)

        i+=1
        t+=dt


#set initial condition
zn = gauss(xx,yy)
un = np.zeros((n + 4, m + 4))
vn = np.zeros((n + 4, m + 4))
#run sim
semi_implicit(zn, un, vn)










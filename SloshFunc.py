from scipy.special import jv
from scipy.optimize import fsolve
import numpy as np
from tqdm import tqdm
from scipy.linalg import eig

def dBesselj(x, n):
    return (jv(n-1, x) - jv(n+1, x)) / 2.0

def dBesselzero(N: int):
    Bzeros = []
    order = 1

    while order <= N:

        if order == 1:
            stepB = 1.8
        else:
            stepB = Bzeros[-1] + np.pi

        Bzero = fsolve(dBesselj, stepB, args = (1))[0]

        if order == 1:
            Bzeros.append(Bzero)
        else: # we have to check if there's duplicated root
            mindiff = np.max(np.abs(np.array(Bzeros) - Bzero))
            if mindiff < 1e-5 * max(Bzeros):
                pass
            else:
                Bzeros.append(Bzero)
        
        order = len(Bzeros) + 1

    # sort the Bzeros array in ascending order
    Bzeros = np.array(Bzeros)

    return np.sort(Bzeros)

eps = 1e-10

def isinside(r, z, r_ic, r_oc):
    return (r >= r_ic(z)) * (r <= r_oc(z))


def Aint(r,z,m,n,M,Bzeros,L,r_ic, r_oc):
    '''
    We only get the integrand for a_mn here. 
    '''

    r = np.clip(r, eps, np.inf)

    if m > M:
        jm = Bzeros[m-M-1]
    
    if n > M:
        jn = Bzeros[n-M-1]

    if m <= M and n <= M :
        I   = (np.abs((2*m-1)*(2*n-1) + 1)* r**(2*n+2*m-4))*r
    elif m <= M and n > M:    
        I   = ((2*m-1)*r**(2*m-2)*jn*dBesselj(jn*r, 1)*np.exp(jn*(z-L)) +\
                r**(2*m-3)*jv(1, jn*r)*np.exp(jn*(z-L)))*r
    elif m > M and n > M:
        I   = (r*jn*jm*dBesselj(jn*r, 1)*dBesselj(jm*r,1) + \
               r**(-1)* jv(1,jn*r)*jv(1,jm*r) + \
               r*jn*jm*jv(1,jn*r)*jv(1, jm*r))*\
               np.exp((jn + jm)*(z-L))

    return I * isinside(r, z, r_ic, r_oc)

def Bint(m,n,M,Bzeros,r_cm):
    '''

    '''
    if m > M:
        jm = Bzeros[m-M-1]
    
    if n > M:
        jn = Bzeros[n-M-1]

    if n <= M and m <= M:
        bmn = (1 - r_cm**(2*n+2*m))/(2*n+2*m)
    elif n > M and m > M and m != n:
        bmn = (jm*r_cm*jv(1,jn*r_cm)*dBesselj(jm*r_cm,1) -\
            jn*r_cm*jv(1,jm*r_cm)*dBesselj(jn*r_cm,1))/\
            (jm**2 - jn**2)
    elif n > M and m > M and m == n:
        bmn = ((jn**2-1)*jv(1,jn)**2 -\
            (jn*r_cm*dBesselj(jn*r_cm,1))**2 -\
            (jn**2*r_cm**2-1)*jv(1,jn*r_cm)**2)/\
            (2*jn**2)
    elif n > M and m == 1:
        bmn = 1/jn**3 * (jn*jv(1,jn) -\
            jn*r_cm*jv(1,jn*r_cm) +\
            jn**2*r_cm**2*dBesselj(jn*r_cm, 1))
        
    elif n > M and m <= M:
        bmn = 1/jn**2*((2*m-1)*jv(1,jn) -\
            r_cm**(2*m-1)*abs((2*m-1)*jv(1,jn*r_cm) -\
            jn*r_cm*dBesselj(jn*r_cm, 1)) - \
            4*m*(m-1)*Bint(m-1,n,M,Bzeros,r_cm))

    return bmn


def simp2D(func,a,b,c,d,NX,NY):
    NX = int(2*np.ceil(NX/2))
    NY = int(2*np.ceil(NY/2))

    hx = (b-a)/NX
    hy = (d-c)/NY

    xg = np.linspace(a, b, NX+1)
    yg = np.linspace(c, d, NY+1)
    xxg, yyg = np.meshgrid(xg, yg)

    U = func(xxg, yyg).T

    s1 = U[0,0] + U[0,NY] + U[NX,0] + U[NX,NY]

    ixo = slice(1,NX,2)
    ixe = slice(2,NX-1,2)
    iyo = slice(1,NY,2)
    iye = slice(2,NY-1,2)

    s2 = 2.0 * (np.sum(U[0, iye]) + np.sum(U[NX, iye]) + np.sum(U[ixe, 0]) + np.sum(U[ixe, NY]))
    s3 = 4.0 * (np.sum(U[0, iyo]) + np.sum(U[NX, iyo]) + np.sum(U[ixo, 0]) + np.sum(U[ixo, NY]))

    s4 = 16.0 * np.sum(U[ixo, iyo]) + 4.0 * np.sum(U[ixe, iye])
    s5 = 8.0 * np.sum(U[ixe, iyo]) + 8.0 * np.sum(U[ixo, iye])

    return (s1+s2+s3+s4+s5)*hx*hy/9.0

def core(N, M, nsteps, bens, e, L, r_ic, r_oc):
    r_min, r_max, z_min = bens[0], bens[1], bens[2]

    Bzeros = dBesselzero(N)

    A = np.zeros((N, N))
    B = np.zeros((N, N))

    for m in tqdm(range(1, N+1)):
        for n in range(m, N+1):
            A[m-1, n-1] = simp2D(lambda r,z: Aint(r,z,m,n,M,Bzeros,L,r_ic,r_oc),
                                 r_min,r_max,z_min,L,nsteps,nsteps)
            B[m-1, n-1] = Bint(m,n,M,Bzeros,e)

    A = (A + A.T) - A * np.eye(N)
    B = (B + B.T) - B * np.eye(N)

    lambda_k, V = eig(A, B)

    return A, B, V, lambda_k, Bzeros

def phi(n, M, R, Z, Bzeros, L_a):
    '''
    n is the mode number, STARTING FROM 1
    '''
    if n <= M:
        return R**(2*n-1)
    else:
        return jv(1, Bzeros[n-M-1]*R) * np.exp(Bzeros[n-M-1]*(Z-L_a))



from scipy.special import jv
from scipy.optimize import fsolve
import numpy as np
from tqdm import tqdm
from scipy.linalg import eig

def read_dict(fname: str):
    data_dict = {}
    with open(fname, "r") as file:
        for line in file:
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            if ("." in val) or ("e" in val):
                data_dict[key] = float(val)
            else:
                data_dict[key] = int(val)

    return data_dict

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

def isinside(r, z, r_ic, r_oc):
    return (r >= r_ic(z)) * (r <= r_oc(z))


def Aint(r,z,m,n,M,Bzeros,L,r_ic, r_oc, bens):
    '''
    We only get the integrand for a_mn here. 
    '''
    eps = 1e-6
    r = np.clip(r, eps, np.inf)
    z = np.clip(z, bens[-2]+eps, bens[-1]-eps)

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

    for m in range(1, N+1):
        for n in range(m, N+1):
            A[m-1, n-1] = simp2D(lambda r,z: Aint(r,z,m,n,M,Bzeros,L,r_ic,r_oc, bens),
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


def derivedparams(rho,flvol,H,c_k,lambda_k,g,L,a,B,s_n,Ph_a,intb,selm):
    '''
    Compute the derived mechanical parameters for the pendulum
    After running this function, the imaginary part of lambda_k
    will be discarded
    '''
    N = s_n.shape[0]
    lambda_k = np.real(lambda_k)

    # the mass of the fluid
    M = flvol*rho

    # gamma that is related to the pendulum
    gamma_k = np.diag(np.pi*a**3/flvol*(c_k.T@B@c_k))

    # b that is related to the pendulum
    b_star_k = (np.pi*a**3)/(flvol*gamma_k) * (c_k.T @ B[0, :])

    # mass of all modes
    M_k = M * lambda_k * gamma_k * (b_star_k**2)

    # the length of the pendulum
    L_k = a / lambda_k

    # insert the mass of the fluid
    # sort the M_k * L_k in the descending mannar
    I = np.flip(np.argsort(M_k * L_k))
    M_k, L_k = M_k[I], L_k[I]

    # only take the firt selm masses into account
    Mtot = np.sum(M_k[0:selm])
    M0 = M - Mtot
    c_k, lambda_k, b_star_k, Ph_a, gamma_k \
        = c_k[:, I], lambda_k[I], b_star_k[I], Ph_a[I], gamma_k[I]
    
    # h parameter
    h_k = (2.0*np.pi*a**3)/(flvol*gamma_k*lambda_k) * (c_k.T @ s_n)
    
    # height coordinate in the z_CM system
    H_k = -H + a*(1.0/lambda_k + h_k/b_star_k)

    # slosh amplitude / pendulum amplitude
    CM_k = Ph_a*b_star_k*lambda_k

    # the height of insert fluid over cg
    Hcg = abs(intb[2])
    H_0 = (M*Hcg - (M_k[0:selm]*(H_k[0:selm]-L_k[0:selm])).sum())/M0

    # Non-dimensional frequency
    K = lambda_k * L/a

    # Natural frequency
    omega_k = np.sqrt(np.abs(lambda_k)*g/a)

    # Characteristic velocity
    Vc = omega_k[0]*a

    # add the fixed mass to the list
    M_k = np.concatenate(([M0], M_k))
    H_k = np.concatenate(([H_0], H_k))
    L_k = np.concatenate(([0], L_k))
    CM_k = np.concatenate(([0], CM_k))

    return K, omega_k, Vc, M_k, L_k, H_k, CM_k, M0, c_k, lambda_k, I


def pendulum2spring(M_k, L_k, g, H_k):
    np.seterr(divide='ignore')
    K_k = M_k*g/L_k
    H_k_spring = H_k - L_k

    return K_k, H_k_spring

def horizspring(startx, starty, length):
    xs, ys = [], []
    xs.append(startx)
    ys.append(starty)

    for n in range(6):
        xs.append(startx + length/12.0 + length/6.0 * n)
        if n % 2 == 0:
            ys.append(starty - length / 6.0)
        else:
            ys.append(starty + length / 6.0)

    xs.append(startx + length)
    ys.append(starty)

    return np.array(xs), np.array(ys)


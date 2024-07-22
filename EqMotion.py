import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.integrate import odeint, quad, quad_vec, simpson

class Integrator:
    def __init__(self, mechanical_profile_dir = "./Dataset", nmodes=3):
        '''
        Integrate the equation of motion of the spring-mass system.
        The mechanical properties (as a function of the LFR) will be
        read in.
        '''
        # transpose the H_k, L_k, and M_k array, the shape after transposition is (N, npoints)
        self.mechanical_profile_dir = mechanical_profile_dir
        self.H_0_data, self.H_k_data = np.load(os.path.join(mechanical_profile_dir, "H_0.npy")),\
                             np.load(os.path.join(mechanical_profile_dir, "H_k.npy")).T[0:nmodes, :]
        self.M_0_data, self.M_k_data = np.load(os.path.join(mechanical_profile_dir, "M_0.npy")),\
                             np.load(os.path.join(mechanical_profile_dir, "M_k.npy")).T[0:nmodes, :]
        self.L_k_data = np.load(os.path.join(mechanical_profile_dir, "L_k.npy")).T[0:nmodes, :]

        # then we will build the interpolators to map the LFR to the mechainical properties
        self.LFR_data = np.load(os.path.join(mechanical_profile_dir, "LFR.npy"))

        # the output has shape (nmodes, npoints)
        self.H_0 = interp1d(self.LFR_data, self.H_0_data)
        self.H_k = interp1d(self.LFR_data, self.H_k_data)
        self.M_0 = interp1d(self.LFR_data, self.M_0_data)
        self.M_k = interp1d(self.LFR_data, self.M_k_data)
        self.L_k = interp1d(self.LFR_data, self.L_k_data)

        # get the upper and the lower bound to safeguard subsequent interpolations
        self.LFR_min, self.LFR_max = np.min(self.LFR_data), np.max(self.LFR_data)
        
        self.nmodes = nmodes

    def dYdt(self, y, t, alpha2, alpha3, theta, d2theta, LFRt):
        '''
        alpha2, alpha3, theta, and d2theta should be functions that give
        the excitation at time t, LFRt is a function that gives the LFR at
        time t

        0:nmodes of y are the locations of the masses
        nmodes: of y are the velocities of the masses
        '''
        LFR = LFRt(t)
        H_kt = self.H_k(LFR)
        M_kt = self.M_k(LFR)
        L_kt = self.L_k(LFR)
        K_kt = M_kt * alpha3(t) / L_kt

        dxdt = y[self.nmodes:]
        dvdt = -(H_kt*d2theta(t) + alpha2(t) + K_kt/M_kt*y[0:self.nmodes] - alpha3(t)*theta(t))

        return np.concatenate((dxdt, dvdt))
        
    def solve(self, y0, t, args):
        sol = odeint(self.dYdt, y0, t, args = args)
        sol_func = interp1d(t, sol.T)

        return sol, sol_func

class Adjoint:
    '''
    This is the adjoint solver to determine the K, M, H 
    parametrically by using the adjoint method.
    '''
    def __init__(self, nmodes, Np, weight = None, t = None, y0 = None, 
                 alpha2 = None, alpha3 = None, 
                 theta = None, LFRt = None,
                 q = None, q_func = None):
        '''
        The user should input the initial guess of the parameters for K, M, H
        '''
        # nmodes is num of modes we want to model, Np is the rank of the polynomials
        # we want to use to approximate 
        self.nmodes, self.Np = nmodes, Np

        self.t = t
        self.y0 = y0
        self.q = q
        self.q_func = q_func
        self.alpha2, self.alpha3, self.theta = alpha2, alpha3, theta
        self.LFRt = LFRt

        if type(weight) != np.ndarray:
            self.weight = np.ones(self.nmodes)
        else:
            assert weight.shape[0] == self.nmodes
            self.weight = weight

    def dYdt_forward(self, y, t, alpha2, alpha3, theta, LFRt):
        '''
        Return the derivative vector of the forward problem

        `theta` should be a float here (we assume the pitching is a constant)
        '''
        LFR = LFRt(t)
        omega = self.omgfunc(LFR)
        dxdt = y[self.nmodes:]
        dvdt = -(alpha2(t) + omega*y[0:self.nmodes] - alpha3(t)*theta)

        return np.concatenate((dxdt, dvdt))
    
    def forward(self, y0, t, alpha2, alpha3, theta, LFRt):
        '''
        Calculate the forward problem. An interpolator of the solution 
        is also returned for later adjoint usage.

        `sol` has shape (2*nmodes, t.shape[0])
        '''
        self.t = t
        self.T = np.max(t)
        self.y0 = y0

        sol = odeint(self.dYdt_forward, y0, t, args=(alpha2,alpha3,theta,LFRt))
        sol_func = interp1d(t, sol.T)

        return sol, sol_func
    
    def dYdtau_backward(self, y, tau, x, q, LFRt):
        '''
        Return the backward derivative (d/dtau = d/d(T-t) = -d/dt) 

        `tau` is the BACKWARD TIME defined by tau = T-t

        `x` is the interpolator of the forward solution;

        `q` is the interpolator of the measured signal in the test (ground truth);

        `LFRt` is the liquid fill ratio interpolator measured in the test

        All the interpolators take FORWARD TIME variable t. 
        '''
        t = max(0.0, self.T-tau)
        LFR = LFRt(t)
        omega = self.omgfunc(LFR)
        q_ = q(t)
        x_ = x(t)
        dlamdt = -omega*y[self.nmodes:] + 2.0 * (q_[0:self.nmodes] - x_[0:self.nmodes]) * self.weight
        # detadt = y[0:self.nmodes] + 2.0 * (q_[self.nmodes:] - x_[self.nmodes:])
        detadt = y[0:self.nmodes] 

        return np.concatenate((dlamdt, detadt))
    
    def backward(self, lam0, tau, x, q, LFRt):
        '''
        After solving the adjoint equation, we flip the lambda
        '''
        lam = odeint(self.dYdtau_backward, lam0, tau, args=(x,q,LFRt))
        lam = np.flip(lam, axis = 0)
        lam_func = interp1d(tau, lam.T)

        return lam, lam_func

    def grad(self, lam, x, lfr):
        '''
        Compute the gradient by numerical integration.
        All inputs are arrays.

        lam: (nt, 2*nmodes)
        x:   (nt, 2*nmodes)
        lfr: (nt,)
        '''
        dFdp = simpson(y=self.grad_integrand(lam,x,lfr),x=self.t)
        return dFdp

    def grad_integrand(self, lam, x, lfr):
        '''
        Compute the integrand (discretized version) for numerical integration.
        All inputs are arrays.

        lam: (nt, 2*nmodes)
        x:   (nt, 2*nmodes)
        lfr: (nt,)
        '''
        x = x.T
        lam = lam.T

        nt = lfr.shape[0]
        domgdp = np.array([lfr**k for k in range(self.Np)]) # Shape: (Np, nt)
        dhdp = np.zeros((self.nmodes, self.nmodes*self.Np, nt))
        dhdp[np.arange(self.nmodes)[:, None], 
             np.arange(self.nmodes*self.Np).reshape(self.nmodes, self.Np), :]\
            = x[0:self.nmodes,np.newaxis,:] * domgdp[np.newaxis,:,:]
        
        dFdp_integrand = np.zeros((self.nmodes*self.Np, nt))
        dFdp_integrand = np.einsum('it,ijt->jt',lam[self.nmodes:,:],dhdp)

        return dFdp_integrand
        
    def objFunc(self, x, q):
        '''
        Calculate the objective function using simpson's integration;
        `x` and `q` should be arrays here.
        '''
        integrand = np.sum(self.weight*(x[:, 0:self.nmodes] - q[:, 0:self.nmodes])**2, axis=1)
        return simpson(y=integrand, x=self.t)
        
    def set_omgfunc(self, p: np.ndarray):
        try:
            self.p_omega = p.reshape(self.nmodes, self.Np)
        except:
            raise Exception("The shape of the input p is not consistent with the solver's setting: Np={:d}, nmodes={:d}".format(self.Np, self.nmodes))

        # update the omega functions
        self.omgfunc = lambda lfr: self.p_omega @ np.array([lfr**k for k in range(self.Np)])
        self.domgfunc = lambda lfr: self.p_omega @ np.array([k*lfr**(max(0, k-1)) for k in range(self.Np)])

    def eval_obj(self, p:np.ndarray):
        '''
        evaluate the objective function given a user specified p.
        The shape of p is (nmodes*Np, )
        '''
        self.set_omgfunc(p)

        self.x, self.x_func = self.forward(self.y0, self.t, self.alpha2, self.alpha3, self.theta, self.LFRt)
        return self.objFunc(self.x, self.q)

    def eval_grad(self):
        '''
        evaluate the gradient at the current parameters stored
        in the object
        '''
        lfr = self.LFRt(self.t)
        self.lam, self.lam_func = self.backward(np.zeros(2*self.nmodes), 
                                                self.t, self.x_func, self.q_func, self.LFRt)
        
        return self.grad(self.lam, self.x, lfr)
    
    def eval_grad_(self, p:np.ndarray):
        '''
        Evaluate the gradient at a given point. In this function, we first
        forward pass the calculation, then we calculate the gradient using
        the adjoint method.
        '''
        objVal = self.eval_obj(p)
        return self.eval_grad()



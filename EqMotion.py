import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.integrate import odeint

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

        return sol





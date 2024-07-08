from scipy.interpolate import interp1d
import scipy.integrate as integrate
import SloshFunc as sf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

class Tank:
    def __init__(self, LFR: float, fname = "./test.txt", settings = "./settings.txt", delimiter = " "):
        '''
        The input z coordinate should be in ascending order.

        In this init function, we read in the contour profile, get the interpolater and the boundarys.
        '''
        contours = np.loadtxt(fname, delimiter=delimiter)
        self.z_ic_original, self.r_ic_original = contours[:, 0], contours[:, 1]
        self.z_oc_original, self.r_oc_original = contours[:, 2], contours[:, 3]

        # drop all nans
        self.z_ic_original = self.z_ic_original[~np.isnan(self.z_ic_original)]
        self.r_ic_original = self.r_ic_original[~np.isnan(self.r_ic_original)]
        self.z_oc_original = self.z_oc_original[~np.isnan(self.z_oc_original)]
        self.r_oc_original = self.r_oc_original[~np.isnan(self.r_oc_original)]

        # get the interpolater
        self.IC_original = interp1d(self.z_ic_original, self.r_ic_original)
        self.OC_original = interp1d(self.z_oc_original, self.r_oc_original)

        # get the boundary
        self.bIC = np.array([np.min(self.r_ic_original), np.max(self.r_ic_original),
                             np.min(self.z_ic_original), np.max(self.z_ic_original)])
        
        self.bOC = np.array([np.min(self.r_oc_original), np.max(self.r_oc_original),
                             np.min(self.z_oc_original), np.max(self.z_oc_original)])
        
        # get the non-dimensional quantities
        self.LFR = LFR
        self.get_geometry(LFR)

        # get the computational settings
        self.settings = sf.read_dict(settings)

    def get_geometry(self, LFR: float):
        '''
        Here we scale the geometry according the fill ratio. The fill ratio is the level of the 
        surface / the entire height of the geometry
        '''
        r_ic_file = self.IC_original # inner contour, unscaled
        r_oc_file = self.OC_original # outer contour, unscaled
        b_ic_file = self.bIC
        b_oc_file = self.bOC

        # Here we do not scale the geometry based on the RFac and ZFac, which is originally done in the MATLAB code
        RFac, ZFac = 1.0, 1.0
        self.RFac, self.ZFac = RFac, ZFac
        b_ens_scaled_origin = np.array([min(b_ic_file[0], b_oc_file[0]) * RFac, max(b_ic_file[1], b_oc_file[1]) * RFac,
                                        max(b_ic_file[2], b_oc_file[2]) * ZFac, min(b_ic_file[3], b_oc_file[3]) * ZFac])
        r_ic_scaled_origin = lambda z: r_ic_file(z/ZFac) * RFac
        r_oc_scaled_origin = lambda z: r_oc_file(z/ZFac) * RFac

        H_tank = LFR * (b_ens_scaled_origin[-1] - b_ens_scaled_origin[-2])
        H_origin = H_tank + b_ens_scaled_origin[-2]

        # perform integration to get the fluid volume and the gravity center
        breaks = np.unique(np.concatenate((self.z_ic_original, self.z_oc_original))) * ZFac
        breaks = np.clip(breaks, b_ens_scaled_origin[-2], H_origin)
        limit = breaks.shape[0] * 5

        fluid_volume = integrate.quad(lambda z: np.pi*((r_oc_scaled_origin(z))**2 - (r_ic_scaled_origin(z))**2), 
                                      b_ens_scaled_origin[-2], H_origin, points = breaks, limit = limit)[0]
        z_CM = integrate.quad(lambda z: z * ((r_oc_scaled_origin(z))**2 - (r_ic_scaled_origin(z))**2), 
                              b_ens_scaled_origin[-2], H_origin, points = breaks, limit = limit)[0]\
              /integrate.quad(lambda z: ((r_oc_scaled_origin(z))**2 - (r_ic_scaled_origin(z))**2),
                              b_ens_scaled_origin[-2], H_origin, points = breaks, limit = limit)[0]
        
        b_ens_scaled_CM = b_ens_scaled_origin - np.array([0.0, 0.0, z_CM, z_CM])
        r_ic_scaled_CM = lambda z: r_ic_scaled_origin(z+z_CM)
        r_oc_scaled_CM = lambda z: r_oc_scaled_origin(z+z_CM)

        # fluid height with respect to the center of mass, measured in the mass-center coordinate
        self.L = H_origin - z_CM

        # scaling parameters:
        self.a = r_oc_scaled_CM(self.L)
        self.epsilon = r_ic_scaled_CM(self.L)
        self.L_a = self.L / self.a

        # the non-dimensional contours (the inputs of the contours are non-dimensional coordinates, the outputs
        # are also non-dimensional)
        self.R_ic_CM = lambda Z: r_ic_scaled_CM(Z * self.a) / self.a
        self.R_oc_CM = lambda Z: r_oc_scaled_CM(Z * self.a) / self.a
        self.IC_scaled_CM = r_ic_scaled_CM
        self.OC_scaled_CM = r_oc_scaled_CM

        # save other quantities
        self.z_CM = z_CM
        self.fluid_volume = fluid_volume
        self.bens = b_ens_scaled_CM
        self.intb = self.bens / self.a
        self.H_tank = H_tank
        self.H_origin = H_origin
 
    def calculate(self):
        self.A, self.B, self.c_k, self.lambda_k, self.Bzeros = \
                                    sf.core(self.settings["N"], self.settings["M"], 
                                    self.settings["nsteps"], self.intb, self.epsilon,
                                    self.L_a, self.R_ic_CM, self.R_oc_CM)
        
        self.compute_mechanical_analogies()

    def compute_mechanical_analogies(self):
        epsilon = self.epsilon
        L_a = self.L_a
        a = self.a
        R_ic = self.R_ic_CM
        R_oc = self.R_oc_CM
        intb = self.intb * a # convert intb to DIMENSIONAL form
        N = self.settings["N"]
        M = self.settings["M"]
        g = self.settings["g"]
        rho = self.settings["rho"]
        flvol = self.fluid_volume
        H = self.H_tank
        selm = self.settings["SpringModeSpinner"]
        lambda_k = self.lambda_k
        c_k = self.c_k
        Bzeros = self.Bzeros


        # Derived parameters
        B = np.zeros((N, N))
        s_n = np.zeros(N)
        phi_a = np.zeros(N)

        # integration break points
        breaks = (np.unique(np.concatenate((self.z_ic_original, self.z_oc_original))) * self.ZFac - self.z_CM) / self.a
        breaks = np.clip(breaks, intb[2]/a, L_a)
        limit = breaks.shape[0] * 5
        
        # calculate the derived parameters
        for n in range(N):
            integrand1 = lambda z: (z-intb[2]/a)*R_oc(z)*sf.phi(n+1,M,R_oc(z),z,Bzeros,L_a)
            integrand2 = lambda z: (z-intb[2]/a)*R_ic(z)*sf.phi(n+1,M,R_ic(z),z,Bzeros,L_a)
            s_n[n] = integrate.quad(integrand1, intb[2]/a, L_a, limit=limit, points=breaks)[0]\
                    -integrate.quad(integrand2, intb[2]/a, L_a, limit=limit, points=breaks)[0]
            phi_a[n] = 0

            for k in range(N):
                phi_a[n] =  phi_a[n] + c_k[k,n]*sf.phi(k+1,M,R_oc(L_a),L_a,Bzeros,L_a)

            for m in range(n, N):
                B[n,m] = sf.Bint(n+1,m+1,M,Bzeros,epsilon)

        # fill the lower-triangle of B
        B = B + B.T - np.eye(N) * B

        self.B_ma, self.s_n, self.phi_a = B, s_n, phi_a
        
        # get the derived pendulum parameters
        K, omega_k, Vc, M_k, L_k, H_k, CM_k, fluid_mass, c_k, lambda_k, I_sort = \
            sf.derivedparams(rho, flvol, H, c_k, lambda_k, g, L_a*a, a, B, s_n, phi_a, intb, selm)

        self.Km, self.omega_k, self.characteristic_velocity, self.M_k, self.L_k, self.H_k, self.CM_k, self.c_k, self.lambda_k, self.I_modesSorting = \
            K, omega_k, Vc, M_k, L_k, H_k, CM_k, c_k, lambda_k, I_sort
        
        self.K_k, self.H_k_spring = sf.pendulum2spring(M_k, L_k, g, H_k)

    def report(self, notebook = True):
        '''
        Report the mechanical characteristics in pandas csv format
        '''
        selm = self.settings["SpringModeSpinner"]
        self.pendulum_table = pd.DataFrame(
            {
                "Mass": self.M_k[0:selm+1],
                "Length": self.L_k[0:selm+1],
                "Height": self.H_k[0:selm+1],
                "CM": self.CM_k[0:selm+1]
            },
            index=["Fixed mass"]+list(range(1, selm+1))
        )

        self.spring_table = pd.DataFrame(
            {
                "Mass": self.M_k[0:selm+1],
                "Height": self.H_k_spring[0:selm+1],
                "K": self.K_k[0:selm+1]
            },
            index=["Fixed mass"]+list(range(1, selm+1))
        )

        self.freq_table = pd.DataFrame(
            {
                "lambda": self.lambda_k,
                "K": self.Km,
                "Omega": self.omega_k
            }
        ).transpose()

        display(self.pendulum_table)
        display(self.spring_table)
        display(self.freq_table)

    def plot_geo(self):
        '''
        Plot the dimensional geometry of the fuel tank
        '''
        bens = self.intb * self.a

        fig = plt.figure(figsize=(3, 6), dpi = 330)

        zv = np.linspace(bens[-2], bens[-1], 501)
        ic = self.IC_scaled_CM(zv)
        oc = self.OC_scaled_CM(zv)
        plt.plot(ic, zv, c = "k", lw = 2)
        plt.plot(oc, zv, c = "k", lw = 2)

        # fluid level
        r_fluid = [self.IC_scaled_CM(self.L), self.OC_scaled_CM(self.L)]
        z_fluid = [self.L, self.L]
        plt.plot(r_fluid, z_fluid, c = "b", lw = 2)

        # center of mass
        plt.scatter([0],[0], marker = "x", c = "green")

        # info
        plt.xlabel("$r$")
        plt.ylabel("$z$")

    def plot_springs(self):
        self.plot_geo()

        number_of_springs_to_plot = self.settings["SpringModeSpinner"]
        H_k_spring = self.H_k_spring
        bens = self.intb * self.a

        for i in range(number_of_springs_to_plot):
            H_spring = H_k_spring[i] + bens[2]

            left = 0.0
            right = self.OC_scaled_CM(H_spring)

            mid = (left + right) / 2.0
            xunit, yunit = abs(bens[1]-bens[0])/10.0, abs(bens[3]-bens[2])/25.0
            springx, springy = sf.horizspring(left+mid/4.0, H_spring, mid / 4.0)

            plt.plot([left, mid/4.0], [H_spring, H_spring], c = "gray", lw = 1.5)
            plt.plot(springx, springy, c="gray", lw = 1.5)
            plt.plot([mid/2.0, right], [H_spring, H_spring], c = "gray", lw = 1.5)
            rect = plt.Rectangle((mid-xunit, H_spring-yunit), xunit*2, yunit*2, facecolor="white", edgecolor = "k", zorder = 20)
            plt.gca().add_patch(rect)
            plt.text(mid-xunit/2.0, H_spring-yunit/3.0, "$m${:d}".format(i), zorder = 25)

    def plot_modes(self, normalize = True, nmodes = None):
        self.plot_geo()
        if nmodes == None:
            number_of_modes_to_plot = self.settings["SpringModeSpinner"]
        else:
            number_of_modes_to_plot = nmodes

        for i_mode in range(number_of_modes_to_plot):
            L, L_a = self.L, self.L_a
            a = self.a
            Bzeros = self.Bzeros
            r_ic_scaled_CM = self.IC_scaled_CM
            r_oc_scaled_CM = self.OC_scaled_CM
            omega_k = self.omega_k
            g = self.settings["g"]
            N = self.settings["N"]
            M = self.settings["M"]
            c_k = self.c_k

            z_vec = np.zeros(200)
            r_vec = np.linspace(r_ic_scaled_CM(L), r_oc_scaled_CM(L), 200)

            for n in range(N):
                z_vec += c_k[n, i_mode]*sf.phi(n+1,M,r_vec/a,L_a,Bzeros,L_a)

            z_vec = z_vec*omega_k[i_mode]*a/g

            if normalize:
                z_vec = z_vec / np.max(np.abs(z_vec))*L/2

            plt.plot(r_vec, z_vec+L, c="r", lw = 1.5)
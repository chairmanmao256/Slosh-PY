# **Slosh-PY**

Slosh-PY is totally based on the Slosh-ML repository (which is built by MATLAB). Slosh-PY is just a Python version of it. We develop this Python version mainly for later possible machine-learning demands.

Developed by Chenyu Wu, during the internship at Orienspace.

## **1.The mechanical parameters of sloshing**

According to the inviscid theory of ideal fluid, we can use spring-mass system or pendulum system to replace the sloshing fluid in the tank without affecting the mecahnical properties of the sloshing fluid. That is to say, under the same excitation, the mechanical analogy and the real sloshing fluid will generate identical response (in terms of the force and the torque produced).

In this repo, we use `Slosh-PY`, which is a Python version of `Slosh-ML`, to calculate the parameters of the mechanical analogy analytically. The key parameters including:

$$
\begin{aligned}
M_k(LFR),H_k(LFR),L_k(LFR)
\end{aligned}
$$

$M_k$ is the mass of the mass-spring system representing the $k^{th}$ sloshing mode, $H_k$ is the height of the mass-spring system measured from the mass-center of the fluid. $L_k$ it the length of the pendulum representing the $k^{th}$ sloshing mode. $LFR$ is the fill ratio of the tank ($LFR\in(0,1)$). The spring constant $K_k$ is then defined as:

$$
K_k(LFR, t)=M_k(LFR)\alpha_3/L_k(LFR)
$$

$\alpha_3$ is the equivalent gravity experienced by the sloshing fluid on the rocket. The equations above mean that both the mass and the location of the mechanical analogy are functions of the fill ratio and the equivalent gravity. On the other hand, the fill ratio and $\alpha_3$ change with time in the flight. Consequently, the mechanical parameters varies continously in the flight.

Assuming that 

## **2.Resolve the mechanical parameters from data**

Suppose that we've obtained the sloshing data from experiment or some other high-fidelity simulation (such as CFD). In this sloshing dataset, the variation of $LFR$, $\alpha_3$, and the excitation (such as $\alpha_2$) with respect to time are also provided. Now we want to use the mechanical analogy in section 1 to build a model from this sloshing data. That is to say, we want to determine the relationship between $M_k, H_k,L_k$ and $LFR$ from this sloshing data. Note that the mechanical analogy described in section 1 that can exactly reconstruct the sloshing data may not exist because of the non-linear and viscous effect in experiment or the simulation. However, we are still interested in to what extent the linear mechanical analogy in section 1 can explain the sloshing data. 

I propose a method similar with the [FIML-Direct](https://www.researchgate.net/profile/Jonathan-Holland-5/publication/330198330_Towards_Integrated_Field_Inversion_and_Machine_Learning_With_Embedded_Neural_Networks_for_RANS_Modeling/links/5c3a42bfa6fdccd6b5a8852e/Towards-Integrated-Field-Inversion-and-Machine-Learning-With-Embedded-Neural-Networks-for-RANS-Modeling.pdf) used in the data-driven turbulence modeling paradigm. I first express the $M_k, L_k$

I've implemented a continuous-adjoint module to extract the mechanical properties from the sloshing data. This is, of course, a data-driven method. However, complex machine learning model is not used in this module. 



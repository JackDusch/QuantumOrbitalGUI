import numpy as np

class StressEnergyTensor:
    def __init__(self, rho_id, rho_cd, four_velocity, metric_tensor, ricci_scalar, gravitational_potential, vector_potential):
        self.rho_id = rho_id
        self.rho_cd = rho_cd
        self.u = four_velocity
        self.g = metric_tensor
        self.R = ricci_scalar
        self.Phi = gravitational_potential
        self.A_mu = vector_potential  # shape (4, Nx, Ny, Nz)

    def compute_F_mn(self, grid_spacing):
        num_components, Nx, Ny, Nz = self.A_mu.shape
        F_mn = np.zeros((4, 4, Nx, Ny, Nz))

        # Spatial derivatives correspond to axes 0 (x), 1 (y), 2 (z) in numpy arrays
        spatial_axes = [0, 1, 2]

        for mu in range(4):
            for nu in range(4):
                if mu != nu:
                    if mu == 0 or nu == 0:
                        # If mu or nu are time indices, handle separately (e.g., zero or specific approach)
                        F_mn[mu, nu] = np.zeros((Nx, Ny, Nz))
                    else:
                        # Adjust indices: mu, nu spatial components range from 1 to 3
                        dA_mu = np.gradient(self.A_mu[mu], grid_spacing, axis=spatial_axes[nu-1])
                        dA_nu = np.gradient(self.A_mu[nu], grid_spacing, axis=spatial_axes[mu-1])
                        F_mn[mu, nu] = dA_mu - dA_nu
        return F_mn

    def compute(self, grid_spacing=1.0):
        F_mn = self.compute_F_mn(grid_spacing)
        T = np.zeros((4, 4))

        for mu in range(4):
            for nu in range(4):
                contraction = np.sum(F_mn[mu] * F_mn[nu])
                energy_density = 0.5 * self.g[mu, nu] * np.sum(F_mn**2) + self.g[mu, nu] * np.sum(self.Phi)
                T[mu, nu] = contraction - energy_density
        return T

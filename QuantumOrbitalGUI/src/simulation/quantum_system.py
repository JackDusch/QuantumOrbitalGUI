import numpy as np
from numba import jit

h_bar = 1.054571817e-34
m_e = 9.10938356e-31
grid_size = 50
dt = 1e-7

 def roll_array_fixed(arr, shift, axis):
    return np.roll(arr, shift=shift, axis=axis)

def laplacian_3d(arr, dx):
    return (
        np.roll(arr,  1, 0) + np.roll(arr, -1, 0) +
        np.roll(arr,  1, 1) + np.roll(arr, -1, 1) +
        np.roll(arr,  1, 2) + np.roll(arr, -1, 2) - 6.0*arr
    ) / (dx*dx)

def update_psi(psi, V, A_mu, dx, dt):
    L = laplacian_3d(psi, dx)
    kinetic = (-h_bar**2/(2*m_e)) * L
    psi_new = psi - (1j*dt/h_bar) * (kinetic + V*psi)
    psi_new *= np.exp(1j * A_mu * dt / h_bar)
    return psi_new

class QuantumSystem:
    def __init__(self, grid_size=50, dt=1e-7):
        self.grid_size = grid_size
        self.dt = dt
        self.dx = 2.0 / grid_size
        x = np.linspace(-1, 1, grid_size)
        self.x, self.y, self.z = np.meshgrid(x, x, x, indexing='ij')
        self.psi = np.exp(-((self.x**2 + self.y**2 + self.z**2) / 0.1)).astype(np.complex128)
        self.V   = 0.5 * m_e * (self.x**2 + self.y**2 + self.z**2)
        self.A_mu= self.x + self.y + self.z

    def step(self):
        self.psi = update_psi(self.psi, self.V, self.A_mu, self.dx, self.dt)




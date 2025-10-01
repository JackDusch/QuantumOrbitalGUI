import numpy as np
from numba import njit
from qutip import basis, Qobj

# Constants
h_bar = 1.054571817e-34
m_e = 9.10938356e-31


@njit
def roll_array_fixed(arr, shift, axis):
    result = np.empty_like(arr)
    if axis == 0:
        result = np.concatenate((arr[-shift:], arr[:-shift]), axis=0)
    elif axis == 1:
        result = np.concatenate((arr[:, -shift:], arr[:, :-shift]), axis=1)
    elif axis == 2:
        result = np.concatenate((arr[:, :, -shift:], arr[:, :, :-shift]), axis=2)
    return result


@njit
def hsv_to_rgb_vectorized(h, s, v):
    h_i = np.floor(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    h_i = h_i.astype(np.int32) % 6

    r = np.where(h_i == 0, v,
        np.where(h_i == 1, q,
        np.where(h_i == 2, p,
        np.where(h_i == 3, p,
        np.where(h_i == 4, t, v)))))

    g = np.where(h_i == 0, t,
        np.where(h_i == 1, v,
        np.where(h_i == 2, v,
        np.where(h_i == 3, q,
        np.where(h_i == 4, p, p)))))

    b = np.where(h_i == 0, p,
        np.where(h_i == 1, p,
        np.where(h_i == 2, t,
        np.where(h_i == 3, v,
        np.where(h_i == 4, v, q)))))

    return r, g, b


@njit
def update_psi(psi, V, A_mu, spacing, dt, h_bar, m):
    laplacian = (
        roll_array_fixed(psi, 1, 0) + roll_array_fixed(psi, -1, 0) +
        roll_array_fixed(psi, 1, 1) + roll_array_fixed(psi, -1, 1) +
        roll_array_fixed(psi, 1, 2) + roll_array_fixed(psi, -1, 2) -
        6 * psi
    ) / spacing[0]**2

    kinetic = (-h_bar**2 / (2 * m)) * laplacian
    potential = V * psi
    phase = np.exp(1j * A_mu * dt / h_bar)

    return (psi - 1j * dt / h_bar * (kinetic + potential)) * phase


class QuantumSystemAdvanced:
    def __init__(self, grid_size=50, dt=1e-7, t_max_steps=1000, use_memmap=False):
        self.grid_size = grid_size
        self.dt = dt
        self.spacing = np.array([2.0 / grid_size] * 3)

        x = np.linspace(-1, 1, grid_size)
        self.x, self.y, self.z = np.meshgrid(x, x, x, indexing='ij')

        self.psi = np.exp(-((self.x**2 + self.y**2 + self.z**2) / 0.1)).astype(np.complex128)
        self.V = 0.5 * m_e * (self.x**2 + self.y**2 + self.z**2)
        self.A_mu = self.x + self.y + self.z

        self.h_bar = h_bar
        self.m = m_e
        self.t_step = 0
        self.t_max_steps = t_max_steps

        self.use_memmap = use_memmap
        if use_memmap:
            self.psi_history = np.memmap("psi_memmap.dat", dtype='complex128', mode='w+',
                                         shape=(t_max_steps, grid_size, grid_size, grid_size))
            self.psi_history[0] = self.psi

        self.H0 = Qobj(np.diag(np.arange(1, 6)))
        self.initial_state = basis(5, 0)

    def step(self):
        self.psi = update_psi(
            self.psi, self.V, self.A_mu, self.spacing, self.dt,
            self.h_bar, self.m
        )
        self.t_step += 1
        if self.use_memmap and self.t_step < self.t_max_steps:
            self.psi_history[self.t_step] = self.psi

    def get_time_slice(self, t_index):
        if self.use_memmap and 0 <= t_index < self.t_max_steps:
            return self.psi_history[t_index]
        raise IndexError("Requested time slice out of bounds.")

    def cleanup(self):
        if self.use_memmap:
            self.psi_history.flush()

    def extract_point_cloud(self, threshold=0.05):
        mask = np.abs(self.psi) > threshold
        x_pts = self.x[mask]
        y_pts = self.y[mask]
        z_pts = self.z[mask]

        magnitude = np.abs(self.psi[mask])
        phase = np.angle(self.psi[mask])
        hue = (phase + np.pi) / (2 * np.pi)

        r, g, b = hsv_to_rgb_vectorized(hue, np.ones_like(hue), magnitude / magnitude.max())

        # Improved alpha using gamma correction
        gamma = 2.2
        alpha = np.power(magnitude / magnitude.max(), 1/gamma)
        alpha = np.clip(alpha * 1.5, 0.0, 1.0)

        return np.stack([x_pts, y_pts, z_pts], axis=1), np.stack([r, g, b, alpha], axis=1)

    def visualize_wave_function(self, psi=None, title="Wave Function"):
        from matplotlib import pyplot as plt
        from matplotlib.colors import hsv_to_rgb

        psi = self.psi if psi is None else psi

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mask = np.abs(psi) > 0.05
        x_pts, y_pts, z_pts = self.x[mask], self.y[mask], self.z[mask]

        magnitude = np.abs(psi[mask])
        phase = np.angle(psi[mask])
        hue = (phase + np.pi) / (2 * np.pi)
        rgb = hsv_to_rgb(np.stack((hue, np.ones_like(hue), magnitude / magnitude.max()), axis=1))

        alpha = np.clip(magnitude / magnitude.max(), 0, 1)
        ax.scatter(x_pts, y_pts, z_pts, c=rgb, alpha=alpha)
        ax.set_title(title)
        plt.show()

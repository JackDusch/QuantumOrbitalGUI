# waveform_transformations.py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from numba import jit

class WaveformTransformations:
    def __init__(self, waveform, wavevector, rgb_waveform, x, y, z):
        self.waveform = np.array(waveform, dtype=np.float64)
        self.wavevector = np.array(wavevector, dtype=np.float64)
        self.rgb_waveform = np.array(rgb_waveform, dtype=np.float64)
        self.x = x
        self.y = y
        self.z = z
        self.grid_points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

        grid_size = waveform.shape[0]
        # In WaveformTransformations.__init__, store a purely 3D amplitude or real-part:
        self.waveform_3d = np.abs(waveform)   # shape Nx×Ny×Nz, for example

        # Then build the interpolator on self.waveform_3d, not the 4D array:
        self.interpolator = RegularGridInterpolator(
        (np.linspace(-1,1,grid_size),
            np.linspace(-1,1,grid_size),
            np.linspace(-1,1,grid_size)),
            self.waveform_3d,
            bounds_error=False,
            fill_value=0
        )

    @staticmethod
    def apply_gamma(waveform, gamma):
        """Raise the waveform to the gamma power (basic gamma correction)."""
        # Guarantee positivity to avoid complex results if waveform < 0
        clipped = np.clip(waveform, a_min=0, a_max=None)
        return clipped ** gamma

    @staticmethod
    def rotation_matrix(yaw, pitch, roll):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R_yaw = np.array([[cy, -sy, 0],
                          [sy,  cy, 0],
                          [0,    0, 1]])
        R_pitch = np.array([[cp, 0, sp],
                            [0,  1, 0],
                            [-sp, 0, cp]])
        R_roll = np.array([[1,  0,   0],
                           [0, cr, -sr],
                           [0, sr,  cr]])

        return R_yaw @ R_pitch @ R_roll

    def integrate_along_ray(self, theta, phi):
        k_hat = np.array([
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
            np.cos(phi) * np.cos(theta)
        ])

        r0 = np.array([0.0, 0.0, 0.0])
        s_vals = np.linspace(0, 2, 100)
        intensities = []

        for s in s_vals:
            r = r0 + s * k_hat
            if np.all(np.abs(r) <= 1.0):
                val = self.interpolator(r)
                # Ensure scalar (sometimes returns array)
                if isinstance(val, np.ndarray):
                    intensities.append(val.item())
                else:
                    intensities.append(val)
            else:
                intensities.append(0.0)

        return np.trapz(intensities, s_vals)

    def rotate_waveform(self, yaw, pitch, roll):
        """Rotate the waveform grid using the provided yaw, pitch, roll (in radians)."""
        R = self.rotation_matrix(yaw, pitch, roll)
        rotated_coords = R @ self.grid_points
        rotated_waveform = self.interpolator(rotated_coords.T).reshape(self.waveform.shape)
        return rotated_waveform

    def apply_fov(self, waveform, fov):
        """Apply a field-of-view scaling to the waveform."""
        scale_factor = np.tan(np.radians(fov) / 2.0)
        return waveform * scale_factor

    def equirectangular_to_stereographic(self, yaw, pitch, roll, fov):
        """An example transformation from an equirectangular image to a stereographic projection."""
        rotated = self.rotate_waveform(yaw, pitch, roll)
        return self.apply_fov(rotated, fov)

    def project_fisheye_to_spherical_impl(self, data,
                                      yaw=0.0, pitch=0.0, roll=0.0,
                                      fov=180.0, radius=0.25,
                                      centerX=0.5, centerY=0.5,
                                      interpolation=0):
        """
        Convert a single fisheye hemisphere in 'data' to an equirectangular projection.
        This implementation assumes that 'data' is a 2D numpy array.
        If 'data' is 3D, the central slice is used.
        """
        # If data is 3D, take the central slice along the first axis.
        if data.ndim != 2:
            data = data[data.shape[0] // 2]

        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        fov_rad = np.radians(fov)

        in_h, in_w = data.shape
        out_h, out_w = in_h, in_w  # for demonstration, same output size
        output = np.zeros((out_h, out_w), dtype=data.dtype)

        # Precompute the rotation matrix for yaw/pitch/roll
        R = self.rotation_matrix(yaw_rad, pitch_rad, roll_rad)

        # For each pixel (u, v) in the output equirectangular image,
        # compute the corresponding spherical angles and then map back to fisheye coords.
        for v_idx in range(out_h):
            phi = np.pi * (0.5 - float(v_idx) / out_h)  # φ ∈ [−π/2, π/2]
            for u_idx in range(out_w):
                theta = 2.0 * np.pi * (float(u_idx) / out_w - 0.5)  # θ ∈ [−π, π]

                # Convert spherical angles to cartesian coordinates
                X = np.cos(phi) * np.cos(theta)
                Y = np.sin(phi)
                Z = np.cos(phi) * np.sin(theta)

                # Rotate coordinates using the rotation matrix R
                v_in = np.array([X, Y, Z])
                v_rot = R @ v_in

                # Use the forward vector (0,0,1) to compute the angle alpha
                forward = np.array([0.0, 0.0, 1.0])
                dot = np.clip(v_rot.dot(forward), -1.0, 1.0)
                alpha = np.arccos(dot)
                if alpha > fov_rad * 0.5:
                    continue

                # Calculate the radial fraction in the fisheye image
                r_frac = alpha / (fov_rad * 0.5)
                x2 = v_rot[0]
                y2 = v_rot[1]
                phi2 = np.arctan2(y2, x2)

                R_pix = radius * in_w * r_frac
                px = centerX * in_w + R_pix * np.cos(phi2)
                py = centerY * in_h + R_pix * np.sin(phi2)

                # Interpolate value from the input data
                if interpolation == 0:
                    px_i = int(round(px))
                    py_i = int(round(py))
                    if 0 <= px_i < in_w and 0 <= py_i < in_h:
                        output[v_idx, u_idx] = data[py_i, px_i]
                else:
                    coords = np.array([[py, px]]).T
                    output[v_idx, u_idx] = map_coordinates(data, coords, order=1, cval=0.0)[0]

        return output

    def project_fisheye_to_spherical(self, data,
                                     yaw=0.0, pitch=0.0, roll=0.0,
                                     fov=180.0, radius=0.25,
                                     frontX=0.5, frontY=0.5,
                                     interpolation=0):
        return self.project_fisheye_to_spherical_impl(
            data,
            yaw, pitch, roll,
            fov, radius,
            centerX=frontX, centerY=frontY,
            interpolation=interpolation
        )

    def project_spherical_to_equirectangular_impl(self, data,
                                                   hfov=90.0, vfov=60.0,
                                                   interpolation=0):
        """
        Convert a rectilinear (normal) image to equirectangular.
        """
        in_h, in_w = data.shape
        out_h, out_w = in_h, in_w
        output = np.zeros((out_h, out_w), dtype=data.dtype)

        hfov_rad = np.radians(hfov)
        vfov_rad = np.radians(vfov)
        fx = in_w / (2.0 * np.tan(hfov_rad * 0.5))
        fy = in_h / (2.0 * np.tan(vfov_rad * 0.5))

        for v_idx in range(out_h):
            phi = np.pi * (0.5 - float(v_idx) / out_h)   # [-π/2, π/2]
            for u_idx in range(out_w):
                theta = 2.0 * np.pi * (float(u_idx) / out_w - 0.5)  # [-π, π]
                X = np.cos(phi) * np.cos(theta)
                Y = np.sin(phi)
                Z = np.cos(phi) * np.sin(theta)
                if abs(Z) < 1e-6:
                    continue
                x_img = fx * (X / Z) + in_w / 2.0
                y_img = fy * (Y / Z) + in_h / 2.0
                if 0 <= x_img < in_w and 0 <= y_img < in_h:
                    if interpolation == 0:
                        ix = int(round(x_img))
                        iy = int(round(y_img))
                        output[v_idx, u_idx] = data[iy, ix]
                    else:
                        coords = np.array([[y_img, x_img]]).T
                        output[v_idx, u_idx] = map_coordinates(data, coords, order=1, cval=0.0)[0]
        return output

    def project_spherical_to_equirectangular(self, data, hfov=90.0, vfov=60.0, interpolation=0):
        """
        If the incoming data is 3D, we reduce it to a single 2D slice. 
        (In a real application, you might do a more sophisticated approach.)
        """
        if data.ndim == 3:
            # Take the central slice along axis=0 (or any axis you want)
            center_slice = data.shape[0] // 2
            data = data[center_slice]

        return self.project_spherical_to_equirectangular_impl(
            data, hfov, vfov, interpolation=interpolation
        )


    def spherical_projection_waveform(self):
        theta_vals = np.linspace(-np.pi, np.pi, 100)
        phi_vals = np.linspace(-np.pi/2, np.pi/2, 100)
        projection = np.zeros((len(theta_vals), len(phi_vals)))
        for i, theta in enumerate(theta_vals):
            for j, phi in enumerate(phi_vals):
                projection[i, j] = self.integrate_along_ray(theta, phi)
        return projection

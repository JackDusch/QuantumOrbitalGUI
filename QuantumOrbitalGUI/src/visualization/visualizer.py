# visualizer.py

import pyqtgraph.opengl as gl
import numpy as np
from pyqtgraph.opengl import GLMeshItem, GLScatterPlotItem, MeshData
from skimage.measure import marching_cubes
from matplotlib.colors import hsv_to_rgb

class Visualizer:
    def __init__(self):
        self.view = gl.GLViewWidget()
        self.view.setWindowTitle('Quantum Orbital Visualization')
        self.view.setCameraPosition(distance=5)

        # Optional grid
        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 2)
        self.view.addItem(self.grid)

        # Items for different visualization modes
        self.mesh_item = GLMeshItem()
        self.mesh_item.setGLOptions('additive')
        self.view.addItem(self.mesh_item)
        
        self.point_cloud_item = GLScatterPlotItem()
        self.point_cloud_item.setGLOptions('additive')
        self.view.addItem(self.point_cloud_item)
        
        # Default to isosurface mode
        self.visualization_mode = "isosurface"  # or "point_cloud"

    def set_visualization_mode(self, mode):
        """Set visualization mode: 'isosurface' or 'point_cloud'"""
        self.visualization_mode = mode
        self.mesh_item.setVisible(mode == "isosurface")
        self.point_cloud_item.setVisible(mode == "point_cloud")

    def update(self, psi, x, y, z, iso_factor=0.2):
        """
        Update visualization based on current mode
        """
        if self.visualization_mode == "isosurface":
            self._update_isosurface(psi, x, y, z, iso_factor)
        else:
            self._update_point_cloud(psi, x, y, z)

    def _update_isosurface(self, psi, x, y, z, iso_factor):
        """Use Marching Cubes to render an isosurface"""
        # 1. Ensure volume is at least 2x2x2
        if psi.ndim < 3:
            psi = np.atleast_3d(psi)
        Nx, Ny, Nz = psi.shape
        if Nx < 2 or Ny < 2 or Nz < 2:
            return

        # 2. Compute magnitude
        magnitude = np.abs(psi)
        max_val = magnitude.max()
        if max_val < 1e-14 or np.isnan(max_val):
            return

        # 3. Determine iso-level
        iso_level = iso_factor * max_val

        # 4. Spacing
        dx = x[1,0,0] - x[0,0,0] if Nx > 1 else 1.0
        dy = y[0,1,0] - y[0,0,0] if Ny > 1 else 1.0
        dz = z[0,0,1] - z[0,0,0] if Nz > 1 else 1.0

        # 5. Run marching_cubes
        try:
            verts, faces, normals, values = marching_cubes(
                volume=magnitude,
                level=iso_level,
                spacing=(dx, dy, dz),
                gradient_direction='descent'
            )
        except Exception as e:
            print(f"Marching cubes error: {e}")
            return

        if verts.size == 0 or faces.size == 0:
            return

        # 6. Color based on phase (like the old visualization)
        # Get phase values at vertices by interpolation
        phase = np.angle(psi)
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (x[:,0,0], y[0,:,0], z[0,0,:]),
            phase,
            bounds_error=False,
            fill_value=0
        )
        vertex_phases = interp(verts)
        
        # Convert phase to HSV colors
        hue = (vertex_phases / (2 * np.pi)) % 1
        saturation = np.ones_like(hue)
        value = np.clip(values / max_val, 0.2, 1.0)  # Use normalized magnitude for brightness
        
        # Convert HSV to RGB
        rgb_colors = hsv_to_rgb(np.column_stack((hue, saturation, value)))
        rgba_colors = np.column_stack((rgb_colors, np.ones_like(hue)))  # Add alpha channel

        # 7. Build mesh data with vertex colors
        mesh_data = MeshData(vertexes=verts, faces=faces)
        mesh_data.setVertexColors(rgba_colors)
        
        self.mesh_item.setMeshData(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,
            drawEdges=False
        )

        # 8. Recenter
        self.mesh_item.resetTransform()
        center_x = (x.max() + x.min()) / 2
        center_y = (y.max() + y.min()) / 2
        center_z = (z.max() + z.min()) / 2
        self.mesh_item.translate(-center_x, -center_y, -center_z)

    def _update_point_cloud(self, psi, x, y, z):
        """Point cloud visualization similar to the old codebase"""
        # Flatten arrays
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        psi_flat = psi.flatten()
        
        # Calculate magnitude and phase
        magnitude = np.abs(psi_flat)
        phase = np.angle(psi_flat)
        
        # Normalize magnitude
        max_magnitude = magnitude.max()
        if max_magnitude < 1e-12:
            return
            
        magnitude_normalized = magnitude / max_magnitude
        
        # Calculate colors (HSV -> RGB)
        hue = (phase / (2 * np.pi)) % 1
        saturation = np.ones_like(hue)
        value = np.clip(magnitude_normalized, 0.2, 1.0)
        
        rgb_colors = hsv_to_rgb(np.column_stack((hue, saturation, value)))
        
        # Calculate alpha (transparency)
        alpha = np.clip(magnitude_normalized, 0, 1)
        
        # Combine into RGBA array
        rgba_colors = np.column_stack((rgb_colors, alpha))
        
        # Create point positions
        points = np.column_stack((x_flat, y_flat, z_flat))
        
        # Set point cloud data
        self.point_cloud_item.setData(
            pos=points,
            color=rgba_colors,
            size=2.0,  # Adjust point size as needed
            pxMode=True  # Points maintain size regardless of zoom
        )

    def show(self):
        self.view.show()
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QComboBox
)
from visualization.waveform_widget import WaveformWidget
from PyQt6.QtCore import QTimer, Qt
import numpy as np

# Import your classes
from simulation.quantum_system_advanced import QuantumSystemAdvanced as QuantumSystem
from simulation.stress_energy import StressEnergyTensor
from transformations.waveform_transformations import WaveformTransformations
from visualization.visualizer import Visualizer

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Orbital Visualization - Russellian Steps")

        # Create the waveform scope widget
        self.waveform_scope = WaveformWidget()

        # Simulation Initialization
        self.simulation = QuantumSystem(grid_size=50, dt=1e-7)

        # Stress-Energy Tensor Example (optional)
        grid_shape = (4, 4, self.simulation.grid_size, self.simulation.grid_size, self.simulation.grid_size)
        self.stress_energy = StressEnergyTensor(
            rho_id=np.zeros(grid_shape),
            rho_cd=np.zeros(grid_shape),
            four_velocity=np.zeros((4, self.simulation.grid_size, self.simulation.grid_size, self.simulation.grid_size)),
            metric_tensor=np.eye(4),
            ricci_scalar=np.zeros(grid_shape),
            gravitational_potential=np.zeros(grid_shape),
            vector_potential=np.zeros((4, self.simulation.grid_size, self.simulation.grid_size, self.simulation.grid_size))
        )

        # Waveform Transformations
        self.transformations = WaveformTransformations(
            waveform=self.simulation.psi.real,
            wavevector=np.gradient(self.simulation.psi.real),
            rgb_waveform=np.abs(self.simulation.psi.real),
            x=self.simulation.x,
            y=self.simulation.y,
            z=self.simulation.z
        )

        # Visualizer
        self.visualizer = Visualizer()

        # Main Layout Setup
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top display area: container widget for visualizer and waveform scope
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.addWidget(self.visualizer.view, stretch=3)
        display_layout.addWidget(self.waveform_scope, stretch=1)
        main_layout.addWidget(display_widget)

        # Bottom control panel
        control_layout = QHBoxLayout()
        self.fisheye_checkbox = QCheckBox("Fisheye → Spherical")
        self.fisheye_checkbox.stateChanged.connect(self.apply_transformations)
        control_layout.addWidget(self.fisheye_checkbox)

        self.equirectangular_checkbox = QCheckBox("Spherical → Equirectangular")
        self.equirectangular_checkbox.stateChanged.connect(self.apply_transformations)
        control_layout.addWidget(self.equirectangular_checkbox)

        self.waveform_checkbox = QCheckBox("Waveform Projection")
        self.waveform_checkbox.stateChanged.connect(self.apply_transformations)
        control_layout.addWidget(self.waveform_checkbox)

        self.wavevector_checkbox = QCheckBox("Wavevector (FFT)")
        self.wavevector_checkbox.stateChanged.connect(self.apply_transformations)
        control_layout.addWidget(self.wavevector_checkbox)

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_button)

        # Additional buttons: Zoom In, Zoom Out, Pan View
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        control_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        control_layout.addWidget(self.zoom_out_button)

        self.pan_button = QPushButton("Pan View")
        self.pan_button.clicked.connect(self.pan_view)
        control_layout.addWidget(self.pan_button)

        main_layout.addLayout(control_layout)

        # Status Label
        self.info_label = QLabel("Simulation Status: Idle")
        main_layout.addWidget(self.info_label)

        # Add this to __init__ after creating other controls
        self.vis_mode_combo = QComboBox()
        self.vis_mode_combo.addItems(["Isosurface", "Point Cloud"])
        self.vis_mode_combo.currentTextChanged.connect(self.change_visualization_mode)
        control_layout.addWidget(QLabel("Visualization Mode:"))
        control_layout.addWidget(self.vis_mode_combo)

        # QTimer for simulation steps
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)

        self.final_waveform = None

    # Add this method to MainWindow class
    def change_visualization_mode(self, mode):
        """Switch between visualization modes"""
        if mode == "Isosurface":
            self.visualizer.set_visualization_mode("isosurface")
        else:
            self.visualizer.set_visualization_mode("point_cloud")
        self.apply_transformations()  # Refresh the display
        
    def start_simulation(self):
        self.timer.start(30)
        self.info_label.setText("Simulation running...")

    def update_simulation(self):
        self.simulation.step()
        self.apply_transformations()
        wave_to_display = self.final_waveform if self.final_waveform is not None else self.simulation.psi
        self.visualizer.update(
            psi=wave_to_display,
            x=self.simulation.x,
            y=self.simulation.y,
            z=self.simulation.z,
            iso_factor=0.3
        )
        self.info_label.setText("Simulation running...")

    def apply_transformations(self):
        data = self.simulation.psi
        if self.fisheye_checkbox.isChecked():
            data = self.transformations.project_fisheye_to_spherical(data)
        if self.equirectangular_checkbox.isChecked():
            data = self.transformations.project_spherical_to_equirectangular(data)
        if self.waveform_checkbox.isChecked():
            data = self.transformations.spherical_projection_waveform()
            self.waveform_scope.setData(data)
        if self.wavevector_checkbox.isChecked():
            if data.ndim == 3:
                data = np.fft.fftshift(np.fft.fftn(data)).real
            else:
                data = np.fft.fftshift(np.fft.fftn(data)).real
            self.waveform_scope.setData(data)
        self.final_waveform = data

    def zoom_in(self):
        current_distance = self.visualizer.view.opts['distance']
        self.visualizer.view.setCameraPosition(distance=current_distance * 0.8)
        self.info_label.setText(f"Zoomed In: {current_distance * 0.8:.2f}")

    def zoom_out(self):
        current_distance = self.visualizer.view.opts['distance']
        self.visualizer.view.setCameraPosition(distance=current_distance * 1.2)
        self.info_label.setText(f"Zoomed Out: {current_distance * 1.2:.2f}")

    def pan_view(self):
        current_azimuth = self.visualizer.view.opts['azimuth']
        new_azimuth = current_azimuth + 15
        self.visualizer.view.orbit(azim=new_azimuth)
        self.info_label.setText(f"Panned: Azimuth = {new_azimuth:.2f}")

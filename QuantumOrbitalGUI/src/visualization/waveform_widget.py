import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage
from PyQt6.QtCore import Qt, QRect

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform_data = None  # 2D array of shape (height, width)

    def setData(self, data):
        """
        data: 2D numpy array representing the waveform or scope intensities.
        """
        self.waveform_data = data
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self.waveform_data is None:
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            painter.end()
            return

        # 🔧 Normalize shape for rendering
        data = self.waveform_data
        if data.ndim == 3:
            # Take center slice for rendering
            data = data[data.shape[0] // 2]
        elif data.ndim == 1:
            # Treat as a single horizontal line
            data = data[np.newaxis, :]

        h, w = data.shape  # ✅ Now guaranteed safe
        scaled = np.clip(data, 0, 255).astype(np.uint8)

        # RGBA construction
        qimg_array = np.zeros((h, w, 4), dtype=np.uint8)
        qimg_array[..., 0:3] = scaled[..., np.newaxis]
        qimg_array[..., 3] = 255

        image = QImage(qimg_array.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        painter.drawImage(self.rect(), image, image.rect())
        painter.end()

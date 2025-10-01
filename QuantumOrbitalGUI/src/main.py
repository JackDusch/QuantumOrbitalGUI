from PyQt6.QtWidgets import QApplication
import sys

from gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    # Initialize the main application window
    window = MainWindow()
    window.setWindowTitle("Quantum Orbital Visualization - Russellian Steps")
    window.resize(800, 600)  # Optional size adjustment
    window.show()

    # Execute the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
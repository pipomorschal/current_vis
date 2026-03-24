from PySide6 import QtWidgets

from signal_visualization_app import MainWindow
from plot_panel_widget import setup_plot_style


def main():
    setup_plot_style()
    app = QtWidgets.QApplication([])
    app.setApplicationName("Reusable Signal Visualization App")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

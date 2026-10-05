from PySide6 import QtWidgets
import faulthandler
import os
from pathlib import Path
import sys
import traceback

from signal_visualization_app import MainWindow
from plot_panel_widget import setup_plot_style


def main():
    log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "SignalVisualizationApp"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Keep this handle alive until Qt and its worker threads have shut down.
    crash_log = open(log_dir / "crash.log", "a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=crash_log, all_threads=True)
    original_hook = sys.excepthook

    def exception_hook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=crash_log)
        original_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = exception_hook
    setup_plot_style()
    app = QtWidgets.QApplication([])
    app.setApplicationName("Signal Visualization App")
    win = MainWindow()
    win.show()
    try:
        return app.exec()
    finally:
        sys.excepthook = original_hook
        faulthandler.disable()
        crash_log.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Keep the UI responsive and instrument ownership held during TEC shutdown."""
from PySide6 import QtCore

from tec_setpoint_process import disable_tec_isolated


class TecShutdownWorker(QtCore.QThread):
    def __init__(self, resource, timeout_ms=5000, parent=None):
        super().__init__(parent)
        self.resource, self.timeout_ms = resource, timeout_ms
        self.error = None

    def run(self):
        try:
            disable_tec_isolated(self.resource, timeout_ms=self.timeout_ms)
        except Exception as exc:
            self.error = str(exc)

    def result_text(self):
        return (f"TEC output OFF could not be verified: {self.error}. Check the controller."
                if self.error else "TEC output OFF verified.")

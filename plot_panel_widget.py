from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg


def setup_plot_style():
    pg.setConfigOptions(antialias=False)


def _nan_safe(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


@dataclass
class PlotDataStore:
    x: np.ndarray
    y: np.ndarray


class ZoomAdaptivePlotPanel(QtWidgets.QWidget):
    view_range_changed = QtCore.Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        self.plot = pg.PlotWidget(title=title)
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.getAxis("left").setPen(pg.mkPen("k"))
        self.plot.getAxis("bottom").setPen(pg.mkPen("k"))
        self.plot.getAxis("left").setTextPen(pg.mkPen("k"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("k"))
        self.plot.setClipToView(True)
        self.plot.setAutoVisible(y=True)

        self._full_data: PlotDataStore | None = None
        self._pen = pg.mkPen("b", width=1.2)
        self._curve = self.plot.plot([], [], pen=self._pen)
        self._additional_curves: list[pg.PlotDataItem] = []
        self._updating = False
        self._first_draw_done = False

        self.min_visible_points = 1000
        self.max_visible_points = 6000

        self._bottom_label = ""
        self._left_label = ""
        self._bottom_units = ""

        self.plot.getViewBox().sigXRangeChanged.connect(self._on_view_range_changed)
        layout.addWidget(self.plot)

    def clear(self):
        self._full_data = None
        self._curve.setData([], [])
        for curve in self._additional_curves:
            curve.setData([], [])
        self._first_draw_done = False

    def set_pen(self, pen):
        self._pen = pen
        self._curve.setPen(pen)

    def set_axis_labels(self, bottom: str = "", left: str = "", bottom_units: str = ""):
        self._bottom_label = bottom
        self._left_label = left
        self._bottom_units = bottom_units
        self.plot.setLabel("bottom", bottom, units=bottom_units if bottom_units else None)
        self.plot.setLabel("left", left)

    def set_data(self, x: np.ndarray, y: np.ndarray, auto_range: bool = True):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x, y = _nan_safe(x, y)

        self._full_data = PlotDataStore(x=x, y=y)
        self.redraw(auto_range=auto_range, initial=True)

    def set_multi_data(self, x: np.ndarray, data_dict: dict[str, np.ndarray], auto_range: bool = True):
        """Plot multiple curves with different colors.
        
        Args:
            x: common x-axis array
            data_dict: dict mapping curve_name -> y_array
            auto_range: whether to auto-fit the view
        """
        x = np.asarray(x, dtype=float)
        
        # Clear additional curves
        for curve in self._additional_curves:
            curve.setData([], [])
        self._additional_curves.clear()
        
        # Use first dataset as primary (for zoom-adaptive logic)
        first_key = next(iter(data_dict.keys())) if data_dict else None
        if first_key is None:
            self._full_data = None
            self._curve.setData([], [])
            return
        
        y_primary = np.asarray(data_dict[first_key], dtype=float)
        x, y_primary = _nan_safe(x, y_primary)
        self._full_data = PlotDataStore(x=x, y=y_primary)
        
        # Plot primary curve
        self.redraw(auto_range=auto_range, initial=True)
        
        # Add additional curves with different colors
        colors = ["r", "g", "c", "m", "y", "orange", "purple", "brown"]
        for idx, (label, y_data) in enumerate(data_dict.items()):
            if label == first_key:
                continue  # Already plotted as primary
            
            y_data = np.asarray(y_data, dtype=float)
            x_safe, y_safe = _nan_safe(x, y_data)
            
            color = colors[(idx - 1) % len(colors)]
            pen = pg.mkPen(color, width=1.2)
            curve = self.plot.plot(x_safe, y_safe, pen=pen, name=label)
            self._additional_curves.append(curve)

    def _visible_window(self) -> tuple[float, float] | None:
        if self._full_data is None or self._full_data.x.size == 0:
            return None
        vr = self.plot.getViewBox().viewRange()
        if not vr or not vr[0]:
            return None
        xmin, xmax = float(vr[0][0]), float(vr[0][1])
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        return xmin, xmax

    def _target_points(self) -> int:
        width = max(1, int(self.plot.width()))
        return int(np.clip(width * 1.5, self.min_visible_points, self.max_visible_points))

    @staticmethod
    def _adaptive_reduce(x: np.ndarray, y: np.ndarray, target_points: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(x)
        if n <= target_points or target_points < 3:
            return x, y

        bins = max(1, target_points // 2)
        edges = np.linspace(0, n, bins + 1, dtype=int)

        xs = []
        ys = []

        for i in range(bins):
            a = edges[i]
            b = edges[i + 1]
            if b <= a:
                continue

            xb = x[a:b]
            yb = y[a:b]
            if xb.size == 0:
                continue

            min_idx = int(np.argmin(yb))
            max_idx = int(np.argmax(yb))
            ordered = (min_idx, max_idx) if min_idx <= max_idx else (max_idx, min_idx)

            for idx in ordered:
                xs.append(xb[idx])
                ys.append(yb[idx])

        if not xs:
            return x, y

        return np.asarray(xs), np.asarray(ys)

    def redraw(self, auto_range: bool = False, initial: bool = False):
        if self._full_data is None:
            self._curve.setData([], [])
            return

        x = self._full_data.x
        y = self._full_data.y

        # On the first draw, show the data directly.
        if not initial:
            window = self._visible_window()
            if window is not None:
                xmin, xmax = window
                left = int(np.searchsorted(x, xmin, side="left"))
                right = int(np.searchsorted(x, xmax, side="right"))
                left = max(0, min(left, len(x)))
                right = max(left, min(right, len(x)))
                x = x[left:right]
                y = y[left:right]

        x, y = self._adaptive_reduce(x, y, self._target_points())

        self._curve.setData(x, y)
        self._curve.setPen(self._pen)

        self.plot.setLabel("bottom", self._bottom_label, units=self._bottom_units if self._bottom_units else None)
        self.plot.setLabel("left", self._left_label)

        if auto_range:
            self.plot.autoRange()

        self._first_draw_done = True

    def _on_view_range_changed(self, *args):
        if self._updating or not self._first_draw_done:
            return
        self._updating = True
        try:
            self.redraw(auto_range=False, initial=False)
            self.view_range_changed.emit()
        finally:
            self._updating = False


PlotPanel = ZoomAdaptivePlotPanel
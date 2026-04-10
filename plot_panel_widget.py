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
        self._full_data_right: PlotDataStore | None = None
        self._pen = pg.mkPen("b", width=1.2)
        self._curve = self.plot.plot([], [], pen=self._pen)
        self._additional_curves: list[pg.PlotDataItem] = []
        self._updating = False
        self._first_draw_done = False

        self._right_view: pg.ViewBox | None = None
        self._right_curve: pg.PlotDataItem | None = None
        self._right_axis_active = False

        self.min_visible_points = 1000
        self.max_visible_points = 6000

        self._bottom_label = ""
        self._left_label = ""
        self._bottom_units = ""

        self.plot.getViewBox().sigXRangeChanged.connect(self._on_view_range_changed)
        self.plot.getViewBox().sigResized.connect(self._sync_right_view_geometry)
        layout.addWidget(self.plot)

    def _ensure_right_axis(self):
        if self._right_view is not None:
            return
        self._right_view = pg.ViewBox()
        self.plot.scene().addItem(self._right_view)
        self.plot.getAxis("right").linkToView(self._right_view)
        self._right_view.setXLink(self.plot.getViewBox())
        self._right_curve = pg.PlotDataItem([], [], pen=pg.mkPen("g", width=1.2))
        self._right_view.addItem(self._right_curve)
        self._sync_right_view_geometry()

    def _sync_right_view_geometry(self):
        if self._right_view is None:
            return
        self._right_view.setGeometry(self.plot.getViewBox().sceneBoundingRect())
        self._right_view.linkedViewChanged(self.plot.getViewBox(), self._right_view.XAxis)

    def _disable_right_axis(self):
        if self._right_view is not None:
            self._right_view.removeItem(self._right_curve)
            self.plot.scene().removeItem(self._right_view)
        self._right_view = None
        self._right_curve = None
        self._right_axis_active = False
        self._full_data_right = None
        self.plot.hideAxis("right")
        self.plot.getAxis("right").setLabel("")

    def clear(self):
        self._full_data = None
        self._curve.setData([], [])
        for curve in self._additional_curves:
            curve.setData([], [])
        self._disable_right_axis()
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
        self._disable_right_axis()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x, y = _nan_safe(x, y)

        self._full_data = PlotDataStore(x=x, y=y)
        self.redraw(auto_range=auto_range, initial=True)

    def set_multi_data(self, x: np.ndarray, data_dict: dict[str, np.ndarray], auto_range: bool = True):
        self._disable_right_axis()
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

    def set_dual_axis_data(
        self,
        x: np.ndarray,
        left_y: np.ndarray,
        right_y: np.ndarray,
        *,
        right_label: str = "",
        right_units: str = "",
        right_pen=None,
        auto_range: bool = True,
    ):
        # Clears old multi-curves because this mode uses one left + one right curve.
        for curve in self._additional_curves:
            curve.setData([], [])
        self._additional_curves.clear()

        self._ensure_right_axis()
        self._right_axis_active = True
        self.plot.showAxis("right")
        self.plot.getAxis("right").setPen(pg.mkPen("k"))
        self.plot.getAxis("right").setTextPen(pg.mkPen("k"))
        self.plot.setLabel("right", right_label, units=right_units if right_units else None)

        if right_pen is None:
            right_pen = pg.mkPen("g", width=1.2)
        if self._right_curve is not None:
            self._right_curve.setPen(right_pen)

        x = np.asarray(x, dtype=float)
        left_y = np.asarray(left_y, dtype=float)
        right_y = np.asarray(right_y, dtype=float)

        x_left, y_left = _nan_safe(x, left_y)
        x_right, y_right = _nan_safe(x, right_y)

        self._full_data = PlotDataStore(x=x_left, y=y_left)
        self._full_data_right = PlotDataStore(x=x_right, y=y_right)
        self.redraw(auto_range=auto_range, initial=True)

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

        if self._right_axis_active and self._full_data_right is not None and self._right_curve is not None:
            xr = self._full_data_right.x
            yr = self._full_data_right.y
            if not initial:
                window = self._visible_window()
                if window is not None:
                    xmin, xmax = window
                    left_r = int(np.searchsorted(xr, xmin, side="left"))
                    right_r = int(np.searchsorted(xr, xmax, side="right"))
                    left_r = max(0, min(left_r, len(xr)))
                    right_r = max(left_r, min(right_r, len(xr)))
                    xr = xr[left_r:right_r]
                    yr = yr[left_r:right_r]
            xr, yr = self._adaptive_reduce(xr, yr, self._target_points())
            self._right_curve.setData(xr, yr)
            self._sync_right_view_geometry()

        if auto_range:
            self.plot.autoRange()
            if self._right_axis_active and self._right_view is not None:
                self._right_view.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                self._right_view.autoRange()

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
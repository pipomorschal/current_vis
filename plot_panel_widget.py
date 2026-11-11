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
        self._additional_data: list[PlotDataStore] = []
        self._metric_views: list[pg.ViewBox] = []
        self._metric_axes: list[pg.AxisItem] = []
        self._legend = None
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
        self.plot.getViewBox().sigResized.connect(self._sync_metric_views)
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
        self._clear_additional_curves()
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
        self._clear_additional_curves()
        self._disable_right_axis()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x, y = _nan_safe(x, y)

        self._full_data = PlotDataStore(x=x, y=y)
        self.redraw(auto_range=auto_range, initial=True)

    def _clear_additional_curves(self):
        for index, curve in enumerate(self._additional_curves):
            if self._metric_views:
                self._metric_views[index].removeItem(curve)
            else:
                self.plot.removeItem(curve)
        for axis, view in zip(self._metric_axes, self._metric_views):
            axis.unlinkFromView()
            self.plot.plotItem.layout.removeItem(axis)
            self.plot.scene().removeItem(axis)
            view.setXLink(None)
            self.plot.scene().removeItem(view)
        self._metric_axes.clear()
        self._metric_views.clear()
        self.plot.getAxis("left").setPen(pg.mkPen("k"))
        self.plot.getAxis("left").setTextPen(pg.mkPen("k"))
        self.plot.getAxis("left").setLabel(self._left_label, color="#000000")
        self._additional_curves.clear()
        self._additional_data.clear()
        if self._legend is not None:
            self._legend.clear()
            self._legend.hide()

    def set_multi_data(self, x: np.ndarray, data_dict: dict[str, np.ndarray], auto_range: bool = True):
        self.set_traces({label: (x, y) for label, y in data_dict.items()}, auto_range=auto_range)

    def _sync_metric_views(self):
        main = self.plot.getViewBox()
        for view in self._metric_views:
            view.setGeometry(main.sceneBoundingRect())
            view.linkedViewChanged(main, view.XAxis)

    def set_traces(self, traces: dict[str, tuple[np.ndarray, np.ndarray]], auto_range: bool = True,
                   *, axis_labels: dict[str, str] | None = None):
        """Display traces with a shared time axis and optional independent y-axes."""
        self._first_draw_done = False
        self._disable_right_axis()
        self._clear_additional_curves()
        if not traces:
            self.clear()
            return
        if self._legend is None:
            self._legend = self.plot.addLegend(labelTextColor="#222222")
        self._legend.show()
        colors = ["#2563eb", "#dc2626", "#15803d", "#9333ea", "#b45309"]
        for index, (label, (x, y)) in enumerate(traces.items()):
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            # Retain missing/undefined values as line breaks (e.g. division by zero).
            finite_time = np.isfinite(x)
            x, y = x[finite_time], y[finite_time]
            y = np.where(np.isfinite(y), y, np.nan)
            data = PlotDataStore(x, y)
            color_index = int(label[2:]) - 1 if label in {"CH1", "CH2", "CH3", "CH4"} else index
            pen = pg.mkPen(colors[color_index % len(colors)], width=1.2)
            if index == 0:
                self._full_data = data
                self._pen = pen
                self._curve.setPen(pen)
                curve = self._curve
                if axis_labels is not None:
                    self._left_label = axis_labels[label]
                    self.plot.getAxis("left").setPen(pen)
                    self.plot.getAxis("left").setTextPen(pen)
                    self.plot.getAxis("left").setLabel(self._left_label, color=pen.color().name())
            else:
                if axis_labels is None:
                    curve = self.plot.plot([], [], pen=pen)
                else:
                    view = pg.ViewBox()
                    self.plot.scene().addItem(view)
                    view.setXLink(self.plot.getViewBox())
                    axis = pg.AxisItem("right")
                    self.plot.plotItem.layout.addItem(axis, 2, 2 + index)
                    axis.linkToView(view)
                    axis.setPen(pen)
                    axis.setTextPen(pen)
                    axis.setLabel(axis_labels[label], color=pen.color().name())
                    curve = pg.PlotDataItem([], [], pen=pen)
                    view.addItem(curve)
                    self._metric_views.append(view)
                    self._metric_axes.append(axis)
                self._additional_curves.append(curve)
                self._additional_data.append(data)
            self._legend.addItem(curve, label)
        self._sync_metric_views()
        self.redraw(auto_range=auto_range, initial=True)

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
        self._clear_additional_curves()

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

            if not np.isfinite(yb).all():
                # Never bridge an undefined interval while reducing a large trace.
                xs.append(xb[0])
                ys.append(np.nan)
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

        self._curve.setData(x, y, connect="finite")
        self._curve.setPen(self._pen)

        for curve, data in zip(self._additional_curves, self._additional_data):
            extra_x, extra_y = data.x, data.y
            window = self._visible_window() if not initial else None
            if window is not None:
                left = int(np.searchsorted(extra_x, window[0], side="left"))
                right = int(np.searchsorted(extra_x, window[1], side="right"))
                extra_x, extra_y = extra_x[left:right], extra_y[left:right]
            extra_x, extra_y = self._adaptive_reduce(extra_x, extra_y, self._target_points())
            curve.setData(extra_x, extra_y, connect="finite")

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
            if self._metric_views:
                # Overlay curves do not contribute to the main ViewBox's bounds.
                times = [data.x for data in [self._full_data, *self._additional_data] if data.x.size]
                if times:
                    self.plot.setXRange(min(float(t.min()) for t in times),
                                        max(float(t.max()) for t in times), padding=0.02)
                self.plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                self.plot.getViewBox().updateAutoRange()
                for view in self._metric_views:
                    view.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                    view.updateAutoRange()
                self._sync_metric_views()
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

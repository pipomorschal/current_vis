import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from signal_visualization_app import MainWindow
from gui_configuration import save_configuration
from ui_controls import CollapsibleSection


def wheel(widget, delta=-120):
    pos = QtCore.QPointF(widget.rect().center())
    global_pos = QtCore.QPointF(widget.mapToGlobal(pos.toPoint()))
    return QtGui.QWheelEvent(pos, global_pos, QtCore.QPoint(), QtCore.QPoint(0, delta),
                            QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
                            QtCore.Qt.ScrollPhase.NoScrollPhase, False)


class UiControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_folding_is_independent_of_enabling_and_preserves_parameters(self):
        content = QtWidgets.QGroupBox("Enable feature")
        content.setCheckable(True)
        content.setChecked(False)
        layout = QtWidgets.QVBoxLayout(content)
        spin = QtWidgets.QSpinBox()
        spin.setValue(42)
        layout.addWidget(spin)
        section = CollapsibleSection("Feature", content)
        section.show()
        try:
            self.app.processEvents()
            self.assertFalse(section.isExpanded())
            self.assertTrue(content.isHidden())
            section.header.click()
            self.app.processEvents()
            self.assertFalse(content.isHidden())
            self.assertFalse(content.isChecked())
            content.setChecked(True)
            self.assertIn("enabled", section.header.text())
            expanded_height = section.sizeHint().height()
            section.header.click()
            self.app.processEvents()
            self.assertTrue(content.isHidden())
            self.assertTrue(content.isChecked())
            self.assertEqual(spin.value(), 42)
            self.assertLess(section.sizeHint().height(), expanded_height)
        finally:
            section.close()

    def test_time_advanced_sections_start_folded_and_stay_folded_across_tabs(self):
        with TemporaryDirectory() as folder:
            w = MainWindow(config_path=Path(folder) / "config.json")
            try:
                for section in (w._logging_section, w._capture_options_section, w._file_options_section,
                                w._tec_sweep_section, w._time_math_section, w._metadata_section,
                                w._time_settings_section):
                    self.assertFalse(section.isExpanded())
                    self.assertTrue(section.content.isHidden())
                w.tabs.setCurrentWidget(w.freq_plot)
                w.tabs.setCurrentWidget(w.time_plot)
                self.assertTrue(w.group_time.isHidden())
                self.assertTrue(w.group_info.isHidden())
                w._logging_section.setExpanded(True)
                w.group_scope_logging.setChecked(True)
                w.spin_scope_log_interval_s.setValue(3.5)
                w._scope_logging_active = True
                w._logging_section.setExpanded(False)
                self.assertTrue(w._scope_logging_active)
                self.assertTrue(w.group_scope_logging.isChecked())
                self.assertEqual(w.spin_scope_log_interval_s.value(), 3.5)
                self.assertIn("enabled", w._logging_section.header.text())
                w._metadata_section.setExpanded(True)
                w.tabs.setCurrentWidget(w.freq_plot)
                w.tabs.setCurrentWidget(w.time_plot)
                self.assertFalse(w.group_info.isHidden())
                w._scope_logging_active = False
                profile = Path(folder) / "profile.json"
                save_configuration(w, profile)
                w.group_scope_logging.setChecked(False)
                self.assertNotIn("enabled", w._logging_section.header.text())
                self.assertTrue(w._load_gui_configuration(profile))
                self.assertIn("enabled", w._logging_section.header.text())
                self.assertFalse(w._logging_section.isExpanded())
            finally:
                w._scope_logging_active = False
                w.close()

    def test_wheel_preserves_focused_and_unfocused_fields_while_scrolling_sidebar(self):
        with TemporaryDirectory() as folder:
            w = MainWindow(config_path=Path(folder) / "config.json")
            try:
                w.resize(1400, 650)
                w._capture_options_section.setExpanded(True)
                w._logging_section.setExpanded(True)
                w.group_scope_logging.setChecked(True)
                with patch.object(w, "_refresh_scope_resources"):
                    w.show()
                    self.app.processEvents()
                fields = (w.spin_scope_timeout_ms, w.spin_scope_log_max_minutes,
                          w.combo_scope_channel, w.combo_scope_log_format)
                bar = w.controls.verticalScrollBar()
                self.assertGreater(bar.maximum(), 200)
                for focus in (False, True):
                    for field in fields:
                        with self.subTest(focus=focus, field=type(field).__name__):
                            field.setFocus() if focus else field.clearFocus()
                            before = field.currentText() if isinstance(field, QtWidgets.QComboBox) else field.value()
                            bar.setValue(0)
                            QtWidgets.QApplication.sendEvent(field, wheel(field))
                            after = field.currentText() if isinstance(field, QtWidgets.QComboBox) else field.value()
                            self.assertEqual(before, after)
                            self.assertGreater(bar.value(), 0)
                # Internal line edits also receive wheel events when editing a number.
                editor = w.spin_scope_timeout_ms.lineEdit()
                before = w.spin_scope_timeout_ms.value()
                bar.setValue(0)
                QtWidgets.QApplication.sendEvent(editor, wheel(editor))
                self.assertEqual(w.spin_scope_timeout_ms.value(), before)
                self.assertGreater(bar.value(), 0)
                # Deliberate keyboard and arrow-button editing remains available.
                w.spin_scope_timeout_ms.setFocus()
                QtTest.QTest.keyClick(w.spin_scope_timeout_ms, QtCore.Qt.Key.Key_Up)
                self.assertEqual(w.spin_scope_timeout_ms.value(), before + w.spin_scope_timeout_ms.singleStep())
            finally:
                w.close()

    def test_guard_covers_other_tabs_and_new_editors_but_allows_open_dropdowns_and_plots(self):
        with TemporaryDirectory() as folder:
            w = MainWindow(config_path=Path(folder) / "config.json")
            try:
                with patch.object(w, "_refresh_scope_resources"):
                    w.show()
                    self.app.processEvents()
                w.tabs.setCurrentWidget(w.tec_sine_widget)
                spin = w.tec_sine_widget.amplitude
                before = spin.value()
                spin.setFocus()
                QtWidgets.QApplication.sendEvent(spin, wheel(spin))
                self.assertEqual(spin.value(), before)
                dynamic = QtWidgets.QComboBox(w.centralWidget())
                dynamic.addItems([str(index) for index in range(100)])
                dynamic.setCurrentIndex(10)
                dynamic.show()
                QtWidgets.QApplication.sendEvent(dynamic, wheel(dynamic))
                self.assertEqual(dynamic.currentIndex(), 10)
                dynamic.setEditable(True)
                QtWidgets.QApplication.sendEvent(dynamic.lineEdit(), wheel(dynamic.lineEdit()))
                self.assertEqual(dynamic.currentIndex(), 10)
                dynamic.showPopup()
                self.app.processEvents()
                self.assertTrue(dynamic.view().isVisible())
                self.assertFalse(w._wheel_value_guard.eventFilter(dynamic.view().viewport(), wheel(dynamic.view().viewport())))
                dynamic.hidePopup()
                self.assertFalse(w._wheel_value_guard.eventFilter(w.time_plot.plot.viewport(), wheel(w.time_plot.plot.viewport())))
                w.close()
                self.assertFalse(w._wheel_value_guard._installed)
                with patch.object(w, "_refresh_scope_resources"):
                    w.show()
                    self.app.processEvents()
                self.assertTrue(w._wheel_value_guard._installed)
                QtWidgets.QApplication.sendEvent(dynamic, wheel(dynamic))
                self.assertEqual(dynamic.currentIndex(), 10)
            finally:
                w.close()


if __name__ == "__main__":
    unittest.main()

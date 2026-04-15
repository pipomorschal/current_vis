from __future__ import annotations

from PySide6 import QtWidgets


class FilePreviewDialog(QtWidgets.QDialog):
    def __init__(self, preview: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Preview")
        self.resize(800, 500)

        self._preview = preview
        self.selected_time_column: str | None = None
        self.selected_amplitude_column: str | None = None
        self._is_hdf5 = preview.get("format") == "hdf5" or bool(preview.get("datasets"))

        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        left_group = QtWidgets.QGroupBox("Datasets" if self._is_hdf5 else "Columns")
        left_layout = QtWidgets.QFormLayout(left_group)
        self.combo_time = QtWidgets.QComboBox()
        self.combo_amp = QtWidgets.QComboBox()
        items = preview.get("datasets", preview.get("columns", []))
        self.combo_time.addItems(items)
        self.combo_amp.addItems(items)
        left_layout.addRow("Time dataset" if self._is_hdf5 else "Time column", self.combo_time)
        left_layout.addRow("Amplitude dataset" if self._is_hdf5 else "Amplitude column", self.combo_amp)
        top.addWidget(left_group, 1)

        right_group = QtWidgets.QGroupBox("Metadata")
        right_layout = QtWidgets.QVBoxLayout(right_group)
        self.table_meta = QtWidgets.QTableWidget(0, 2)
        self.table_meta.setHorizontalHeaderLabels(["Key", "Value"])
        self.table_meta.horizontalHeader().setStretchLastSection(True)
        self.table_meta.verticalHeader().setVisible(False)
        self.table_meta.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.table_meta)
        top.addWidget(right_group, 2)

        info_group = QtWidgets.QGroupBox("File Info")
        info_layout = QtWidgets.QFormLayout(info_group)
        info_layout.addRow("File", QtWidgets.QLabel(str(preview.get("file_path", ""))))
        info_layout.addRow("Data rows", QtWidgets.QLabel(str(preview.get("row_count", 0))))
        layout.addWidget(info_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._fill_metadata_table(preview.get("metadata", {}))

        columns = preview.get("datasets", preview.get("columns", []))
        upper = [c.upper() for c in columns]
        if "TIME" in upper:
            self.combo_time.setCurrentIndex(upper.index("TIME"))
        elif "TIME" not in upper and columns:
            self.combo_time.setCurrentIndex(0)

        preferred_amp = ["AMPLITUDE", "SIGNAL", "CH1"]
        for candidate in preferred_amp:
            if candidate in upper:
                self.combo_amp.setCurrentIndex(upper.index(candidate))
                break
        else:
            if len(columns) > 1:
                self.combo_amp.setCurrentIndex(1)

    def _fill_metadata_table(self, metadata: dict[str, str]):
        self.table_meta.setRowCount(len(metadata))
        for row, (key, value) in enumerate(metadata.items()):
            self.table_meta.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
            self.table_meta.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))

    def accept(self):
        self.selected_time_column = self.combo_time.currentText().strip() or None
        self.selected_amplitude_column = self.combo_amp.currentText().strip() or None
        super().accept()

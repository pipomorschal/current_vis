from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class FilePreviewDialog(QtWidgets.QDialog):
    def __init__(self, preview: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Preview")
        self.resize(800, 500)

        self._preview = preview
        self.selected_time_column: str | None = None
        self.selected_amplitude_column: str | None = None
        self.selected_additional_columns: tuple[str, ...] = ()
        self._is_hdf5 = preview.get("format") == "hdf5" or bool(preview.get("datasets"))

        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        left_group = QtWidgets.QGroupBox("Datasets" if self._is_hdf5 else "Columns")
        left_layout = QtWidgets.QFormLayout(left_group)
        self.combo_time = QtWidgets.QComboBox()
        self.combo_amp = QtWidgets.QComboBox()
        items = preview.get("datasets", preview.get("columns", []))
        self.combo_time.addItems(preview.get("time_datasets") or items)
        self.combo_amp.addItems(preview.get("measurement_datasets") or items)
        left_layout.addRow("Time dataset" if self._is_hdf5 else "Time column", self.combo_time)
        left_layout.addRow("Amplitude dataset" if self._is_hdf5 else "Amplitude column", self.combo_amp)
        left_column = QtWidgets.QVBoxLayout()
        left_column.addWidget(left_group)
        self.measurement_list = None
        if (self._is_hdf5 and preview.get("measurement_datasets")) or (not self._is_hdf5 and not any(
                key in preview.get("metadata", {}) for key in ("captured_channel_data", "reference_channel"))):
            self.resize(950, 650)
            group = QtWidgets.QGroupBox("Additional measurements to plot")
            group_layout = QtWidgets.QVBoxLayout(group)
            self.measurement_search = QtWidgets.QLineEdit()
            self.measurement_search.setPlaceholderText("Find a measurement…")
            group_layout.addWidget(self.measurement_search)
            self.measurement_list = QtWidgets.QListWidget()
            counts = preview.get("populated_counts", {})
            for name in preview.get("measurement_datasets", items):
                count = counts.get(name)
                label = preview.get("dataset_labels", {}).get(name, name)
                text = label if label == name else f"{label} ({name})"
                item = QtWidgets.QListWidgetItem(text + (f"  ({count:,} values)" if count is not None else ""))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked if name in preview.get("default_additional_datasets", ())
                                   else QtCore.Qt.CheckState.Unchecked)
                if count == 0:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.measurement_list.addItem(item)
            group_layout.addWidget(self.measurement_list)
            self.measurement_search.textChanged.connect(self._filter_measurements)
            self.combo_time.currentTextChanged.connect(self._filter_measurements)
            self.combo_amp.currentTextChanged.connect(self._filter_measurements)
            if self._is_hdf5:
                self.combo_amp.currentTextChanged.connect(self._select_measurement_time)
            left_column.addWidget(group, 1)
            for combo in (self.combo_time, self.combo_amp):
                combo.setEditable(True)
                combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
                combo.completer().setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
                combo.completer().setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        top.addLayout(left_column, 2)

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
        if preview.get("time_origin_utc"):
            info_layout.addRow("Time origin (UTC)", QtWidgets.QLabel(preview["time_origin_utc"]))
            info_layout.addRow("Time axis", QtWidgets.QLabel("Elapsed seconds from the first timestamp"))
        layout.addWidget(info_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._fill_metadata_table(preview.get("metadata", {}))

        columns = [self.combo_time.itemText(i) for i in range(self.combo_time.count())]
        upper = [c.upper() for c in columns]
        if "TIME" in upper:
            self.combo_time.setCurrentIndex(upper.index("TIME"))
        elif "TIME" not in upper and columns:
            self.combo_time.setCurrentIndex(0)

        columns = [self.combo_amp.itemText(i) for i in range(self.combo_amp.count())]
        upper = [c.upper() for c in columns]
        preferred_amp = ["AMPLITUDE", "SIGNAL", "CH1"]
        for candidate in preferred_amp:
            if candidate in upper:
                self.combo_amp.setCurrentIndex(upper.index(candidate))
                break
        else:
            if len(columns) > 1:
                self.combo_amp.setCurrentIndex(1)
        if self._is_hdf5:
            default_time = preview.get("default_time_dataset")
            default_amplitude = preview.get("default_amplitude_dataset")
            if default_time is not None and self.combo_time.findText(default_time) >= 0:
                self.combo_time.setCurrentText(default_time)
            if default_amplitude is not None and self.combo_amp.findText(default_amplitude) >= 0:
                self.combo_amp.setCurrentText(default_amplitude)
            self._select_measurement_time()
        self._filter_measurements()

    def _select_measurement_time(self, *_):
        spec = self._preview.get("measurements", {}).get(self.combo_amp.currentText(), {})
        axis = spec.get("time_dataset")
        if axis is not None and self.combo_time.findText(axis) >= 0:
            self.combo_time.setCurrentText(axis)

    def _filter_measurements(self, *_):
        if self.measurement_list is None:
            return
        text = self.measurement_search.text().casefold()
        excluded = {self.combo_time.currentText(), self.combo_amp.currentText()}
        for i in range(self.measurement_list.count()):
            item = self.measurement_list.item(i)
            name = item.data(QtCore.Qt.ItemDataRole.UserRole)
            item.setHidden(name in excluded or text not in item.text().casefold())

    def _fill_metadata_table(self, metadata: dict[str, str]):
        self.table_meta.setRowCount(len(metadata))
        for row, (key, value) in enumerate(metadata.items()):
            self.table_meta.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
            self.table_meta.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))

    def accept(self):
        self.selected_time_column = self.combo_time.currentText().strip() or None
        self.selected_amplitude_column = self.combo_amp.currentText().strip() or None
        if self.measurement_list is not None:
            self.selected_additional_columns = tuple(
                item.data(QtCore.Qt.ItemDataRole.UserRole)
                for item in (self.measurement_list.item(i) for i in range(self.measurement_list.count()))
                if item.checkState() == QtCore.Qt.CheckState.Checked
                and item.data(QtCore.Qt.ItemDataRole.UserRole) not in {self.selected_time_column, self.selected_amplitude_column})
        super().accept()

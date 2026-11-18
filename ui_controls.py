"""Compact settings sections and mouse-wheel protection for parameter editors."""
from PySide6 import QtCore, QtGui, QtWidgets


class CollapsibleSection(QtWidgets.QWidget):
    """Folding changes visibility independently of an optional enable checkbox."""

    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.content = content
        self._title = title
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.header = QtWidgets.QToolButton()
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setAutoRaise(True)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        font = self.header.font()
        font.setBold(True)
        self.header.setFont(font)
        layout.addWidget(self.header)
        layout.addWidget(content)
        if isinstance(content, QtWidgets.QGroupBox):
            if content.isCheckable():
                content.toggled.connect(self._update_title)
            else:
                content.setTitle("")
        self.header.toggled.connect(self._set_expanded)
        self._update_title()
        self._set_expanded(False)

    def _update_title(self, *_):
        enabled = (isinstance(self.content, QtWidgets.QGroupBox)
                   and self.content.isCheckable() and self.content.isChecked())
        self.header.setText(self._title + (" (enabled)" if enabled else ""))

    def _set_expanded(self, expanded):
        self.header.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)
        self.header.setToolTip(("Hide " if expanded else "Show ") + self._title)
        self.content.setVisible(expanded)
        self.updateGeometry()

    def setExpanded(self, expanded):
        self.header.setChecked(expanded)

    def isExpanded(self):
        return self.header.isChecked()


class WheelValueGuard(QtCore.QObject):
    """Route editor wheel gestures to their scroll area, even when focused.

    One application filter also covers editors created later in this window.
    An explicitly opened combo popup keeps normal list scrolling.
    """

    def __init__(self, window):
        super().__init__(window)
        self._installed = False
        self.install()

    def install(self):
        if not self._installed:
            QtWidgets.QApplication.instance().installEventFilter(self)
            self._installed = True

    def uninstall(self):
        if self._installed:
            QtWidgets.QApplication.instance().removeEventFilter(self)
            self._installed = False

    def eventFilter(self, watched, event):
        if event.type() != QtCore.QEvent.Type.Wheel or not isinstance(watched, QtWidgets.QWidget):
            return False
        editor = watched
        while editor is not None and not isinstance(editor, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)):
            editor = editor.parentWidget()
        owner = self.parent()
        if editor is None or not isinstance(owner, QtWidgets.QWidget) or not owner.isAncestorOf(editor):
            return False
        if isinstance(editor, QtWidgets.QComboBox) and editor.view().isVisible():
            return False

        container = editor.parentWidget()
        while container is not None and not isinstance(container, QtWidgets.QAbstractScrollArea):
            container = container.parentWidget()
        if container is not None:
            viewport = container.viewport()
            position = viewport.mapFromGlobal(event.globalPosition().toPoint())
            forwarded = QtGui.QWheelEvent(
                QtCore.QPointF(position), event.globalPosition(), event.pixelDelta(), event.angleDelta(),
                event.buttons(), event.modifiers(), event.phase(), event.inverted(), event.source(), event.device())
            QtWidgets.QApplication.sendEvent(viewport, forwarded)
        event.accept()
        return True

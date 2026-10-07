"""A lightweight Snake game, independent of acquisition and analysis."""
import random

from PySide6 import QtCore, QtGui, QtWidgets


class SnakeBoard(QtWidgets.QWidget):
    changed = QtCore.Signal()
    size = 20

    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(140)
        self.timer.timeout.connect(self.advance)
        self.reset()

    def reset(self):
        self.timer.stop()
        self.snake = [(10, 10), (9, 10), (8, 10)]
        self.direction = self.pending_direction = (1, 0)
        self.score = 0
        self.finished = False
        self.message = "Press Start to play"
        self.place_food()
        self.changed.emit()
        self.update()

    def place_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.snake]
        self.food = random.choice(free) if free else None

    def toggle_pause(self):
        if self.finished:
            self.reset()
        if self.timer.isActive():
            self.pause()
        else:
            self.message = ""
            self.timer.start()
            self.changed.emit()
            self.update()
        self.setFocus()

    def pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.message = "Paused"
            self.changed.emit()
            self.update()

    def hideEvent(self, event):
        self.pause()
        super().hideEvent(event)

    def keyPressEvent(self, event):
        keys = QtCore.Qt.Key
        directions = {keys.Key_Left: (-1, 0), keys.Key_A: (-1, 0),
                      keys.Key_Right: (1, 0), keys.Key_D: (1, 0),
                      keys.Key_Up: (0, -1), keys.Key_W: (0, -1),
                      keys.Key_Down: (0, 1), keys.Key_S: (0, 1)}
        if event.key() in directions:
            candidate = directions[event.key()]
            # Accept one turn per tick; rapid keys cannot reverse into the body.
            if self.pending_direction == self.direction and candidate != tuple(-v for v in self.direction):
                self.pending_direction = candidate
        elif event.key() == keys.Key_Space and not event.isAutoRepeat():
            self.toggle_pause()
        elif event.key() == keys.Key_R:
            self.reset()
            self.toggle_pause()
        else:
            super().keyPressEvent(event)

    def advance(self):
        if self.finished:
            return
        self.direction = self.pending_direction
        x, y = self.snake[0]
        dx, dy = self.direction
        head = ((x + dx) % self.size, (y + dy) % self.size)
        eating = head == self.food
        body = self.snake if eating else self.snake[:-1]
        if head in body:
            self.timer.stop()
            self.finished = True
            self.message = "Game over — press R to restart"
        else:
            self.snake.insert(0, head)
            if eating:
                self.score += 1
                self.place_food()
                if self.food is None:
                    self.timer.stop()
                    self.finished = True
                    self.message = "You win! Press R to restart"
            else:
                self.snake.pop()
        self.changed.emit()
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        cell = min(self.width(), self.height()) / self.size
        left = (self.width() - cell * self.size) / 2
        top = (self.height() - cell * self.size) / 2
        field = QtCore.QRectF(left, top, cell * self.size, cell * self.size)
        painter.fillRect(field, QtGui.QColor("#18212b"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#657586"), 1))
        painter.drawRect(field.adjusted(0.5, 0.5, -0.5, -0.5))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for index, (x, y) in enumerate(self.snake):
            painter.setBrush(QtGui.QColor("#b4ed79" if index == 0 else "#65b86a"))
            painter.drawRoundedRect(QtCore.QRectF(left + x * cell + 1, top + y * cell + 1,
                                                 cell - 2, cell - 2), 3, 3)
        if self.food is not None:
            x, y = self.food
            painter.setBrush(QtGui.QColor("#ff866b"))
            painter.drawEllipse(QtCore.QRectF(left + x * cell + 3, top + y * cell + 3,
                                             cell - 6, cell - 6))
        if self.message:
            painter.fillRect(field, QtGui.QColor(0, 0, 0, 160))
            painter.setPen(QtGui.QColor("white"))
            painter.drawText(field, QtCore.Qt.AlignmentFlag.AlignCenter, self.message)


class MiniGameWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.board = SnakeBoard()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.board)
        self.control_panel = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(self.control_panel)
        title = QtWidgets.QLabel("Snake")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        controls.addWidget(title)
        instructions = QtWidgets.QLabel("Collect the red dots. Avoid your tail.\n"
                                      "Cross an edge to appear on the opposite side.\n\n"
                                      "Arrow keys / WASD: move\nSpace: start or pause\nR: restart\n\n"
                                      "The game pauses when you leave this tab.")
        instructions.setWordWrap(True)
        controls.addWidget(instructions)
        self.score_label = QtWidgets.QLabel()
        controls.addWidget(self.score_label)
        self.play_button = QtWidgets.QPushButton("Start")
        self.play_button.clicked.connect(self.board.toggle_pause)
        controls.addWidget(self.play_button)
        restart = QtWidgets.QPushButton("Restart")
        restart.clicked.connect(self.restart)
        controls.addWidget(restart)
        controls.addStretch()
        self.board.changed.connect(self.update_controls)
        self.update_controls()

    def restart(self):
        self.board.reset()
        self.board.toggle_pause()

    def update_controls(self):
        self.score_label.setText(f"Score: {self.board.score}")
        self.play_button.setText("Pause" if self.board.timer.isActive() else
                                 "Play again" if self.board.finished else "Start / Resume")

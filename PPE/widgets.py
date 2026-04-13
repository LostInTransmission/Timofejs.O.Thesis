from PyQt6 import QtWidgets, QtCore
import pyqtgraph.opengl as gl

class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title=""):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.toggle_button = QtWidgets.QToolButton(self)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.toggle_button.clicked.connect(self.on_pressed)
        self.layout.addWidget(self.toggle_button)
        self.content_area = QtWidgets.QFrame(self)
        self.content_layout = QtWidgets.QVBoxLayout(self.content_area)
        self.layout.addWidget(self.content_area)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)
        self.content_area.setVisible(checked)

class SyncGLViewWidget(gl.GLViewWidget):
    cameraChanged = QtCore.pyqtSignal(dict)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.syncing = False
        
    def mouseMoveEvent(self, ev):
        if ev.buttons() & QtCore.Qt.MouseButton.LeftButton:
            return
        super().mouseMoveEvent(ev)
        if ev.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            if not self.syncing:
                self.cameraChanged.emit(self.opts)
                
    def wheelEvent(self, ev):
        super().wheelEvent(ev)
        if not self.syncing:
            self.cameraChanged.emit(self.opts)
            
    def sync_from(self, opts):
        self.syncing = True
        self.opts['center'] = opts['center']
        self.opts['distance'] = opts['distance']
        self.opts['fov'] = opts['fov']
        self.update()
        self.syncing = False
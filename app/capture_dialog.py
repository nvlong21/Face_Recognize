
from PyQt5 import QtCore, QtGui, QtWidgets

import pickle
from libs.canvas import Canvas
from functools import partial
from libs.utils import *
from libs.label_dialog import LabelDialog
class CaptureDialog(QDialog):
    def __init__(self, parent = None):
        QDialog.__init__(self, parent)

        self.parent = parent
        self.data = None
        self.init_ui()

    def init_ui(self):
        self.centralWidget = QtWidgets.QFrame(self)
        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.setGeometry(QtCore.QRect(0, 0, 600, 600))
        self.centralWidget.setFrameShape(QtWidgets.QFrame.Box)

        self.frame_2 = QtWidgets.QFrame(self.centralWidget)
        self.frame_2.setObjectName("frame_2")
        self.frame_2.setGeometry(QtCore.QRect(10, 10, 580, 580))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.canvas= Canvas(self.frame_2)
        
        self.canvas.setGeometry(QtCore.QRect(0, 0, 580, 580))
        self.canvas.setEnabled(True)
        self.canvas.setFocus(True)
        self.canvas.setDrawingShapeToSquare(False)
        self.canvas.restoreCursor()
        self.canvas.mode = self.canvas.CREATE
        self.image = None
        self.canvas.show()
        self.canvas.newShape.connect(self.new_shape)
    def new_shape(self):
        """Pop-up and give focus to the label editor.
        position MUST be in global coordinates.
        """
        BB = QDialogButtonBox

        x1, y1, x2, y2 = int(self.canvas.line[0].x()), int(self.canvas.line[0].y()), int(self.canvas.line[1].x()), int(self.canvas.line[1].y())
        image = self.image[y1:y2, x1:x2]
        self.labelDialog = LabelDialog(parent=self, image = image)
        
        self.labelDialog.show()
        
       
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.utils import newIcon, labelValidator

BB = QDialogButtonBox
from libs.constants import FACE_BANK
from datetime import datetime
import os,cv2
class LabelDialog(QDialog):

    def __init__(self, text="Enter name label", parent=None, image=None):
        super(LabelDialog, self).__init__(parent)

        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setValidator(labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        self.image = image
        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        bb.accepted.connect(self.handleOK)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)
    # def load(self):
    #     self.setDefault()
    #     try:
    #         if os.path.exists(self.path):
    #             with open(self.path, 'rb') as f:
    #                 self.data = pickle.load(f)     
    #         else:
    #             self.setDefault()
    #         return True
    #     except:
    #         print('Loading setting failed')
    #     return False

    def handleOK(self):
        Name = self.edit.text()
        try:
            if self.edit.text().trimmed():
                self.accept()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            if self.edit.text().strip():
                self.accept()

        qm = QMessageBox
        g_image_name = '%s/%s/%s'%(FACE_BANK, Name, Name) + '_' + datetime.now().strftime('%Y%m%d_%H:%M:%S')+ '.jpg'
        if os.path.exists('%s/%s'%(FACE_BANK, Name)):
            qm().question(self, '', "%s is exists! Are you continue?"%Name, qm.Yes | qm.No)
            if qm.Yes:
                cv2.imwrite(g_image_name, self.image)
        else:
            os.mkdir('%s/%s'%(FACE_BANK, Name))
            cv2.imwrite(g_image_name, self.image)

    def validate(self):
        try:
            if self.edit.text().trimmed():
                self.accept()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            if self.edit.text().strip():
                self.accept()

    def postProcess(self):
        try:
            self.edit.setText(self.edit.text().trimmed())
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            self.edit.setText(self.edit.text())

    def popUp(self, text='', move=True):
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        self.edit.setFocus(Qt.PopupFocusReason)
        if move:
            self.move(QCursor.pos())
        return self.edit.text() if self.exec_() else None

    def listItemClick(self, tQListWidgetItem):
        try:
            text = tQListWidgetItem.text().trimmed()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            text = tQListWidgetItem.text().strip()
        self.edit.setText(text)

    def listItemDoubleClick(self, tQListWidgetItem):
        self.listItemClick(tQListWidgetItem)
        self.validate()
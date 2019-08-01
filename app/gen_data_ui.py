import cv2
import os
import sys
import numpy as np
from datetime import datetime
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi
class TakePic(QDialog):
    def __init__(self,parent = None):
        QDialog.__init__(self, parent)
        self.parent = parent
        self.id_cam = 0
        self.init_ui()

    def init_ui(self):
        self.centralWidget = QtWidgets.QFrame(self)
        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.setGeometry(QtCore.QRect(0, 0, 600, 700))
        # self.centralWidget.setFrameShape(QtWidgets.QFrame.Box)

        self.frame_2 = QtWidgets.QFrame(self.centralWidget)
        self.frame_2.setObjectName("frame_2")
        self.frame_2.setGeometry(QtCore.QRect(10, 10, 580, 580))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.video_feed = QtWidgets.QLabel(self.frame_2)
        self.video_feed.setGeometry(QtCore.QRect(0, 0, 580, 580))
        self.video_feed.setObjectName("video_feed")
        self.video_feed.raise_()

        self.frame3 = QtWidgets.QFrame(self.centralWidget)
        self.frame3.setGeometry(QtCore.QRect(10, 590, 690, 110))
        self.frame3.setObjectName("frame3")

        self.label_3 = QLabel('Name: ', self.frame3)
        self.label_3.setGeometry(QtCore.QRect(10, 15, 71, 25))

        self.img_name = QTextEdit(self.frame3)
        self.img_name.setGeometry(QtCore.QRect(91, 15, 287, 25))


        self.btn_close = QtWidgets.QPushButton(self.frame3)
        self.btn_close.clicked.connect(self.start_timer)
        self.btn_close.setGeometry(QtCore.QRect(50, 55, 87, 31))
        self.btn_close.setObjectName("btn_close")
        self.btn_close.setText('Close')

        self.btn_ok = QtWidgets.QPushButton(self.frame3)
        self.btn_ok.setGeometry(QtCore.QRect(450, 55, 87, 31))
        self.btn_ok.setObjectName("btn_ok")
        self.btn_ok.setText('OK')
        self.btn_take_pic = QtWidgets.QPushButton(self.frame3)
        self.btn_take_pic.setGeometry(QtCore.QRect(250, 55, 87, 31))
        self.btn_take_pic.setObjectName("btn_take_pic")
        self.btn_take_pic.setText('Capture')
        self.allow_capture = True
        self.images = None

    def run_video_capture(self):
        self.capture = cv2.VideoCapture(self.id_cam)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.start_timer)
        self.timer.start(50)
    def start_timer(self):
        self.ret, frame = self.capture.read()
        if self.ret and self.allow_capture:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(600,600))
            self.image = frame.copy()
        # frame = cv2.flip(frame, 1)
            image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], 
                           frame.strides[0], QtGui.QImage.Format_RGB888)
            self.video_feed.setPixmap(QtGui.QPixmap.fromImage(image))


        

    def stop_timer(self):       # stop timer or come out of the loop.
        self.timer.stop()
        self.ret = False
        self.capture.release()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
            return fileName

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore
import cv2

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        max_size = QtCore.QSize(sizeObject.width(), sizeObject.height())
        min_size = max_size/2
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1071, 680)
        MainWindow.setMinimumSize(1071, 680)
        MainWindow.setMaximumSize(1071, 680)
        MainWindow.setBaseSize(1071, 680)
        MainWindow.setGeometry(QtCore.QRect(0, 0, 1071, 680))
        self.centralWidget = QtWidgets.QFrame(MainWindow)

        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.setGeometry(QtCore.QRect(0, 0, 1071, 680))
        self.centralWidget.setFrameShape(QtWidgets.QFrame.Box)

        self.frame = QtWidgets.QFrame(self.centralWidget)
        self.frame.setGeometry(QtCore.QRect(190, 10, 870, 560))
        self.frame.setObjectName("frame")

        self.video_feed = QtWidgets.QLabel(self.frame)
        self.video_feed.setGeometry(QtCore.QRect(0, 0, 870, 560))
        self.video_feed.setObjectName("video_feed")
        self.video_feed.raise_()

        self.frame_2 = QtWidgets.QFrame(self.centralWidget)
        self.frame_2.setObjectName("frame_2")
        self.frame_2.setGeometry(QtCore.QRect(10, 10, 171, 560))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)

        self.groupBox_2 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 150, 191))
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_2.setTitle('ACTION')

        self.generate_dataset_btn = QtWidgets.QPushButton('Gender Facebank', self.groupBox_2)
        self.generate_dataset_btn.setGeometry(QtCore.QRect(10, 30, 130, 29))
        self.generate_dataset_btn.setObjectName("generate database")

        self.train_model_btn = QtWidgets.QPushButton('Train', self.groupBox_2)
        self.train_model_btn.setGeometry(QtCore.QRect(10, 70, 130, 29))
        self.train_model_btn.setObjectName("train_model_btn")

        self.recognize_face_btn = QtWidgets.QPushButton( 'Recognize', self.groupBox_2)
        self.recognize_face_btn.setGeometry(QtCore.QRect(10, 110, 130, 29))
        self.recognize_face_btn.setObjectName("recognize_face_btn")

        self.capture_btn = QtWidgets.QPushButton( 'Capture', self.groupBox_2)
        self.capture_btn.setGeometry(QtCore.QRect(10, 150, 130, 29))
        self.capture_btn.setObjectName("recognize_face_btn")


        self.groupBox_1 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_1.setGeometry(QtCore.QRect(10, 220, 150, 261))
        self.groupBox_1.setObjectName("groupBox_1")
        self.groupBox_1.setTitle('SETTING')

        self.mtcnn = QtWidgets.QLabel(self.groupBox_1)
        self.mtcnn.setGeometry(QtCore.QRect(10, 30, 130, 22))
        self.mtcnn.setObjectName("mtcnn_label")
        self.mtcnn.setText('MTCNN: ')

        self.model = QtWidgets.QLabel(self.groupBox_1)
        self.model.setGeometry(QtCore.QRect(10, 60, 130, 30))
        self.model.setObjectName("model_label")
        self.model.setText('MODEL: None')

        self.face_limit = QtWidgets.QLabel(self.groupBox_1)
        self.face_limit.setGeometry(QtCore.QRect(10, 90, 130, 30))
        self.face_limit.setObjectName("face_limit_label")
        self.face_limit.setText('FACE LIMIT: 5')

        self.thresold = QtWidgets.QLabel(self.groupBox_1)
        self.thresold.setGeometry(QtCore.QRect(10, 120, 130, 30))
        self.thresold.setObjectName("thresold_label")
        self.thresold.setText('THRESOLD: 1.2')

        self.min_face_size = QtWidgets.QLabel(self.groupBox_1)
        self.min_face_size.setGeometry(QtCore.QRect(10, 150, 130, 30))
        self.min_face_size.setObjectName("thresold_label")
        self.min_face_size.setText('MIN FACE SIZE: 30')

        self.video_recording_btn = QtWidgets.QPushButton(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay), 'Record', self.centralWidget)
        self.video_recording_btn.setGeometry(QtCore.QRect(35, 510, 87, 31))
        self.video_recording_btn.setObjectName("video_recording_btn")
        self.video_recording_btn.setCheckable(True)


        self.frame3 = QtWidgets.QFrame(self.centralWidget)
        self.frame3.setGeometry(QtCore.QRect(10, 580, 1051, 61))
        self.frame3.setObjectName("frame3")
        self.frame3.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame3.setFrameShadow(QtWidgets.QFrame.Plain)

        self.progress_bar_recognize = QtWidgets.QProgressBar(self.frame3)
        self.progress_bar_recognize.setGeometry(QtCore.QRect(490, 20, 118, 23))
        self.progress_bar_recognize.setObjectName("progress_bar_recognize")
        self.progress_bar_recognize.setValue(0)
        self.progress_bar_recognize.setFormat('%p%')

        self.label_3 = QtWidgets.QLabel(self.frame3)
        self.label_3.setGeometry(QtCore.QRect(420, 20, 71, 21))
        self.label_3.setText('Confidence')

        self.progress_bar_recognize = QtWidgets.QProgressBar(self.frame3)
        self.progress_bar_recognize.setGeometry(QtCore.QRect(490, 20, 118, 23))
        self.progress_bar_recognize.setObjectName("progress_bar_recognize")
        self.progress_bar_recognize.setValue(0)
        self.progress_bar_recognize.setFormat('%p%')

        self.label = QtWidgets.QLabel(self.frame3)
        self.label.setGeometry(QtCore.QRect(230, 20, 54, 21))
        self.label.setText('Trainined')

        self.progress_bar_train = QtWidgets.QProgressBar(self.frame3)
        self.progress_bar_train.setGeometry(QtCore.QRect(290, 20, 118, 23))
        self.progress_bar_train.setObjectName("progress_bar_train")
        self.progress_bar_train.setValue(0)
        self.progress_bar_train.setFormat('%p%')

        self.label_2 = QtWidgets.QLabel(self.frame3)
        self.label_2.setGeometry(QtCore.QRect(30, 20, 54, 21))
        self.label_2.setText('Generated')

        self.progress_bar_generate = QtWidgets.QProgressBar(self.frame3)
        self.progress_bar_generate.setGeometry(QtCore.QRect(100, 20, 118, 23))
        self.progress_bar_generate.setObjectName("progress_bar_generate")
        self.progress_bar_generate.setValue(0)
        self.progress_bar_generate.setFormat('%p%')


        self.exit_btn = QtWidgets.QPushButton(self.frame3)
        self.exit_btn.clicked.connect(self.on_click)
        self.exit_btn.setGeometry(QtCore.QRect(700, 15, 87, 31))
        self.exit_btn.setObjectName("exit_btn")
        self.exit_btn.setText('Exit')



        self.adv_setting = QtWidgets.QPushButton(self.frame3)

        self.adv_setting.setGeometry(QtCore.QRect(900, 15, 87, 31))
        self.adv_setting.setObjectName("adv_setting")
        self.adv_setting.setText('Setting')

        self.centralWidget.raise_()
        MainWindow.setCentralWidget(self.centralWidget)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")

        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")

        MainWindow.setStatusBar(self.statusBar)
        self.menu_bar = QtWidgets.QMenuBar(MainWindow)
        self.menu_bar.setGeometry(QtCore.QRect(0, 0, 1071, 32))
        self.menu_bar.setDefaultUp(False)

        self.capture = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

    def open(self):
        fileNames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Files")
        print(fileNames)
        if len(fileNames)> 0:
            self.capture = cv2.VideoCapture(fileNames[0])
            self.timer.start(30)
        self.statusBar.showMessage("run video",3000)
    def clicked(bool):
        self.close()


    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QtGui.QImage.Format_RGB888)
        self.lblCamView.setPixmap(QtGui.QPixmap.fromImage(image))
    def re_change_source(self, link):
        self.capture = cv2.VideoCapture(link)
        self.timer.start(30)

    @QtCore.pyqtSlot()
    def on_click(self):
        self.close()

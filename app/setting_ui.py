import cv2
import os
import sys
import numpy as np
from easydict import EasyDict as edict
from datetime import datetime
import torch
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi
import pickle
from libs.constants import WORK_PATH, FACE_BANK
from utils.config import get_config
class Setting(QDialog):
    def __init__(self,parent = None):
        QDialog.__init__(self, parent)

        self.path = os.path.join(WORK_PATH, 'libs/setting/setting.pkl')
        self.path_default =os.path.join(WORK_PATH, 'libs/setting/default.pkl')
        self.parent = parent
        self.data = None
        self.changeDefault = False
        self.device = 'gpu'

        self.init_ui()
        self.load_form()

    def init_ui(self):
        self.centralWidget = QtWidgets.QFrame(self)
        self.centralWidget.setObjectName("centralWidget")
        self.centralWidget.setGeometry(QtCore.QRect(0, 0, 600, 600))
        # self.centralWidget.setFrameShape(QtWidgets.QFrame.Box)

        self.frame_2 = QtWidgets.QFrame(self.centralWidget)
        self.frame_2.setObjectName("frame_2")
        self.frame_2.setGeometry(QtCore.QRect(10, 10, 171, 480))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)

        self.groupBox_3 = QGroupBox(self.frame_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 151, 150, 131))
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_3.setTitle('MODEL')

        self.rect_radio_group = QButtonGroup(self.groupBox_3)
        self.model_ir_se_radio = QRadioButton(self.groupBox_3)
        self.model_ir_se_radio.setGeometry(QtCore.QRect(20, 90, 120, 30))
        self.model_ir_se_radio.setObjectName("model_ir_se")
        self.model_ir_se_radio.setText('ir_se')

        self.model_ir_radio = QRadioButton(self.groupBox_3)
        self.model_ir_radio.setGeometry(QtCore.QRect(20, 60, 120, 30))
        self.model_ir_radio.setObjectName("model_ir")
        self.model_ir_radio.setText('model_ir')

        self.mobile_radio = QRadioButton(self.groupBox_3)
        self.mobile_radio.setGeometry(QtCore.QRect(20, 30, 120, 30))
        self.mobile_radio.setObjectName("Mobile")
        self.mobile_radio.setText('Mobile')
        
        self.rect_radio_group.addButton(self.model_ir_se_radio)
        self.rect_radio_group.addButton(self.model_ir_radio)
        self.rect_radio_group.addButton(self.mobile_radio)


        self.groupBox_1 = QGroupBox(self.frame_2)
        self.groupBox_1.setGeometry(QtCore.QRect(10, 20, 150, 91))
        self.groupBox_1.setObjectName("groupBox_1")
        self.groupBox_1.setTitle('ACTION')

        self.algo_radio_group = QButtonGroup()
       
        self.mtcnn1_radio = QRadioButton(self.groupBox_1)
        self.mtcnn1_radio.setGeometry(QtCore.QRect(10, 30, 120, 22))
        self.mtcnn1_radio.setObjectName("mtcnn1_radio")
        self.mtcnn1_radio.setText('MTCNN 1')
        self.mtcnn1_radio.buttonGroup = self.algo_radio_group

        self.mtcnn2_radio = QRadioButton(self.groupBox_1)
        self.mtcnn2_radio.setGeometry(QtCore.QRect(10, 60, 120, 22))
        self.mtcnn2_radio.setObjectName("mtcnn2_radio")
        self.mtcnn2_radio.setText('MTCNN 2')
        self.mtcnn2_radio.buttonGroup = self.algo_radio_group
        self.algo_radio_group.addButton(self.mtcnn1_radio)
        self.algo_radio_group.addButton(self.mtcnn2_radio)

        self.frame_1 = QtWidgets.QFrame(self.centralWidget)
        self.frame_1.setObjectName("frame_2")
        self.frame_1.setGeometry(QtCore.QRect(190, 10, 400, 480))
        self.frame_1.setFrameShape(QtWidgets.QFrame.Box)


        self.groupBox_2 = QGroupBox(self.frame_1)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 20, 380, 450))
        self.groupBox_2.setObjectName("groupBox_1")
        self.groupBox_2.setTitle('ACTION')

        self.label_1 = QLabel('video source: ', self.groupBox_2)
        self.label_1.setGeometry(QtCore.QRect(10, 30, 120, 22))

        self.video_source = QTextEdit(self.groupBox_2)
        self.video_source.setGeometry(QtCore.QRect(121, 30, 250, 22))

        self.label_3 = QLabel('Thresold: ', self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(10, 60, 120, 22))

        self.threshold_text = QTextEdit(self.groupBox_2)
        self.threshold_text.setGeometry(QtCore.QRect(121, 60, 250, 22))

        self.label_2 = QLabel('Log path: ', self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 90, 120, 22))

        self.Log_path = QTextEdit(self.groupBox_2)
        self.Log_path.setGeometry(QtCore.QRect(121, 90, 250, 22))

        self.label_4 = QLabel('Face Bank: ', self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(10, 120, 120, 22))

        self.facebank_path = QtWidgets.QPushButton(self.groupBox_2)
        self.facebank_path.setObjectName("facebank_btn")
        self.facebank_path.setGeometry(QtCore.QRect(121, 120, 180, 22))

        self.update_facebank = QtWidgets.QPushButton(self.groupBox_2)
        self.update_facebank.setObjectName("update_facebank_btn")
        self.update_facebank.setGeometry(QtCore.QRect(310, 120, 60, 22))
        self.update_facebank.setText("Update")

        self.label_5 = QLabel('Device: ', self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(10, 150, 120, 22))

        self.cbDevice = QComboBox(self.groupBox_2)
        self.cbDevice.addItem("cpu")
        self.cbDevice.addItem("cuda")
        self.cbDevice.setGeometry(QtCore.QRect(121, 150, 250, 22))
        self.cbDevice.activated[str].connect(self.change_device)

        self.frame3 = QtWidgets.QFrame(self.centralWidget)
        self.frame3.setGeometry(QtCore.QRect(10, 490, 600, 100))
        self.frame3.setObjectName("frame3")

        self.btn_close = QtWidgets.QPushButton(self.frame3)
        self.btn_close.setGeometry(QtCore.QRect(25, 15, 87, 31))
        self.btn_close.setObjectName("btn_close")
        self.btn_close.setText('Close')

        self.btn_default = QtWidgets.QPushButton(self.frame3)
        self.btn_default.setGeometry(QtCore.QRect(220, 15, 87, 31))
        self.btn_default.setObjectName("btn_default")
        self.btn_default.setText('Default')

        self.btn_ok = QtWidgets.QPushButton(self.frame3)
        self.btn_ok.setGeometry(QtCore.QRect(400, 15, 87, 31))
        self.btn_ok.setObjectName("btn_ok")
        self.btn_ok.setText('OK')
        
    def closeEvent(self, event):
        pass

    def change_device(self, device):
        hight_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dev = 'cuda' if (str(device) =='cuda' and str(hight_device) !='cpu') else 'cpu'
        if device == dev:
            pass
        else:
            QMessageBox().about(self, "warring", "\"%s\" device not found, device change to: \"%s\""%(device, dev))
        self.device = torch.device(dev)


    def load_form(self):
        self.load()
        self.re_load()

    def re_load(self):
        if self.get('use_mtcnn'):
            self.mtcnn1_radio.setChecked(True)
        else:
            self.mtcnn2_radio.setChecked(True)

        if self.get('net_mode')=='mobi':
            self.mobile_radio.setChecked(True)
            self.model_ir_radio.setChecked(False)
            self.model_ir_se_radio.setChecked(False)
        elif self.get('net_mode') == 'ir':
            self.mobile_radio.setChecked(False)
            self.model_ir_radio.setChecked(True)
            self.model_ir_se_radio.setChecked(False)
        else:
            self.mobile_radio.setChecked(False)
            self.model_ir_radio.setChecked(False)
            self.model_ir_se_radio.setChecked(True)
        self.facebank_path.setText(self.get('facebank_path'))
        self.threshold_text.setText(str(self.get('threshold')))

        self.Log_path.setText(self.get('log_path'))
        self.video_source.setText('Webcam' if self.get('video_source') ==0 else self.get('video_source'))
        self.device = self.get('device')
        self.cbDevice.setCurrentText(str(self.device))

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default

    def change_setting(self):
        conf = self.data
        if self.model_ir_radio.isChecked():
            conf.net_mode = 'ir' # or 'ir'
        elif self.model_ir_se_radio.isChecked():
            conf.net_mode = 'ir_se' # or 'ir'
        else:
            conf.net_mode = 'mobi'
        
        conf.threshold = 1.2 if self.threshold_text.toPlainText() == '' else float(self.threshold_text.toPlainText())
        conf.use_mtcnn = True if self.mtcnn1_radio.isChecked() else False
        conf.face_limit = 5 
        conf.min_face_size = 30 
        conf.use_tensor = True
        conf.work_path = os.path.dirname(WORK_PATH)
        conf.model_path =  os.path.dirname(WORK_PATH)
        conf.facebank_path = FACE_BANK
        conf.use_mobile_facenet = True if self.mobile_radio.isChecked() else False
        conf.video_source = 0 if self.video_source.toPlainText() == 'Webcam' else self.video_source.toPlainText()
        conf.device = self.device 
        self.data = conf
        return self.data

    def get_config(self):
        return self.data
        
    def save(self):
        success = False
        if self.path:
            with open(self.path, 'wb') as f:
                pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
                success = True
        return success

    def load(self):
        success = False
        try:
            if os.path.exists(self.path):
                with open(self.path, 'rb') as f:
                    self.data = pickle.load(f)     
            success =True
        except:
            if self.set_default():
                QMessageBox().about(self, "warring",'Loading setting failed, change setting to default')
            else:
                QMessageBox().about(self, "error",'Loading setting failed!')
        return success

    def reset(self):
        success = False
        if os.path.exists(self.path):
            if self.set_default(has_result = True):
                self.re_load()
                os.remove(self.path)
                print('Remove setting pkl file ${0}'.format(self.path))
                self.save()
                success =True
        return success

    def set_default(self, has_result = False):
        success = False
        try:
            if os.path.exists(self.path_default):
                with open(self.path_default, 'rb') as f:
                    self.data = pickle.load(f)
            else:
                conf = get_config(net_mode = 'ir_se', use_mtcnn = 1, threshold = 1.2) 
                self.data =  conf 
            success = True
        except:
            QMessageBox().about(self, "error",'Loading setting failed!')
        if has_result:
            return success

    def openFileNameDialog(self, for_att):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.pth)", options=options)
        if fileName:
            if os.path.isfile(fileName):
                return fileName
        return self.get(for_att)
import cv2
import os
import sys
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
import window_ui
sys.path.insert(0, "..")
from api import face_recognize
from libs.constants import FACE_BANK
from utils.config import get_config
from utils.utils import draw_box_name
from PIL import Image
import torch
from gen_data_ui import TakePic
from setting_ui import Setting
from capture_dialog import CaptureDialog

class AUFR(QMainWindow, window_ui.Ui_MainWindow):        # Main application 
    """Main Class"""
    def __init__(self):
        super(AUFR, self).__init__()
        self.setupUi(self)
        try:
            self.setting = Setting(self)
            conf = self.setting.get_config()
            self.threshold = conf.threshold
            self.model_name = 'mobile face' if conf.net_mode is None else conf.net_mode
            self.net_mode = conf.net_mode
            self.use_mtcnn = True if conf.use_mtcnn else False
            self.camera_id = conf.video_source
            self.face_recognize = face_recognize(conf)
        except:
            pass
            conf = get_config(net_mode = 'ir_se', use_mtcnn = True, threshold = 1.25)
            self.threshold = conf.threshold
            self.model_name = 'mobile face' if conf.net_mode is None else conf.net_mode
            self.net_mode = conf.net_mode
            self.use_mtcnn = True if conf.use_mtcnn else False
            self.camera_id = conf.video_source
            self.face_recognize = face_recognize(conf)
        
        self.targets , self.names = self.face_recognize.get_facebank()
        self.has_targ = len(self.targets)>0
        # Variables
        self.camera_id = 'video.mp4' # can also be a url of Video
        self.dataset_per_subject = 50
        self.ret = False
        self.trained_model = 0
        self.image = cv2.imread("icon/default.jpg", 1)
        self.modified_image = self.image.copy()
        self.reload()
        self.display()
        # Actions 
        self.generate_dataset_btn.setCheckable(True)
        # self.train_model_btn.setCheckable(True)
        self.recognize_face_btn.setCheckable(True)
        # Menu
        self.about_menu = self.menu_bar.addAction("About")
        self.help_menu = self.menu_bar.addAction("Help")
        self.about_menu.triggered.connect(self.about_info)
        self.help_menu.triggered.connect(self.help_info)

        # Algorithms
        self.generate_dataset_btn.clicked.connect(self.pressedGendataButton)

        self.recognize_face_btn.clicked.connect(self.recognize)

        self.video_recording_btn.clicked.connect(self.save_video)

        self.adv_setting.clicked.connect(self.pressedSettingsButton)

        self.capture_btn.clicked.connect(self.captureDialog_show)
        
        if not os.path.exists(FACE_BANK):
            os.mkdir(FACE_BANK)

        self.createMenus()
        # Recognizers

    def reload(self):
        self.mtcnn.setText('MTCNN: '+ str(self.use_mtcnn))
        self.model.setText('MODEL: '+ str(self.model_name).upper())
        self.thresold.setText('THRESOLD: 1.2')

    def start_timer(self):      # start the timeer for execution.
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QtCore.QTimer()
        if self.generate_dataset_btn.isChecked():
            pass
            # self.timer.timeout.connect(self.save_dataset)
        elif self.recognize_face_btn.isChecked():
            self.timer.timeout.connect(self.update_image)
        self.timer.start(5)

    def stop_timer(self):       # stop timer or come out of the loop.
        self.timer.stop()
        self.ret = False
        if self.capture is not None:
            self.capture.release()
    def captureDialog_show(self):
        capture_dialog = captureDialog(self)
        # capture_img = cv2.resize(self.image, (580, 580))
        frame = cv2.resize(self.image, (580, 440)) 
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QtGui.QImage.Format_RGB888)
        capture_dialog.image = frame
        capture_dialog.canvas.loadPixmap(QPixmap.fromImage(image))
        capture_dialog.show()

    def pressedSettingsButton(self):
        new_settingup = Setting(self)
        def handleOK():
            new_settingup.btn_ok.setEnabled(False)
            new_settingup.facebank_path.setEnabled(False)
            new_settingup.btn_close.setEnabled(False)
            new_settingup.btn_default.setEnabled(False)
            new_settingup.change_setting()
            st = new_settingup.get_config()
            if new_settingup.save():
                self.model_name = 'mobile face' if st.net_mode is 'mobi' else st.net_mode
                if self.use_mtcnn != st.use_mtcnn:
                    self.use_mtcnn = st.use_mtcnn
                    self.face_recognize.setup(new_settingup.data)
                if self.net_mode != st.net_mode:
                	try:
	                    self.net_mode =st.net_mode
	                    self.face_recognize.setup(new_settingup.data)
	                    self.face_recognize.update_facebank()
	                    self.targets , self.names =self.face_recognize.get_facebank()
	                except:
	                	QMessageBox().about(self, "Error","can't update facebank!")
                if self.camera_id != st.video_source:
                    self.camera_id = st.video_source
                    self.stop_timer()
                    self.start_timer()

                self.setting = new_settingup
                self.reload()
                self.has_targ = len(self.targets)>0
            new_settingup.btn_ok.setEnabled(True)
            new_settingup.facebank_path.setEnabled(True)
            new_settingup.btn_close.setEnabled(True)
            new_settingup.btn_default.setEnabled(True)
            
            new_settingup.close()
        
        def handleClose():
            new_settingup.close()

        def handleChangeFaceBank():
            new_settingup.facebank_path.setText(new_settingup.openFileNameDialog(for_att= 'facebank_path'))
            self.has_targ = len(self.targets)>0

        def handleUpFaceBank():
            new_settingup.btn_ok.setEnabled(False)
            new_settingup.facebank_path.setEnabled(False)
            new_settingup.btn_close.setEnabled(False)
            new_settingup.btn_default.setEnabled(False)
            self.targets , self.names = self.face_recognize.update_facebank()

            self.has_targ = len(self.targets)>0
            new_settingup.btn_ok.setEnabled(True)
            new_settingup.facebank_path.setEnabled(True)
            new_settingup.btn_close.setEnabled(True)
            new_settingup.btn_default.setEnabled(True)

        def handleChangeDefault():
            if new_settingup.reset():
                QMessageBox().about(self, "OK", "Change setting default success!")
            else:
                QMessageBox().about(self, "Error", "Change setting default error!")
        new_settingup.show()
        new_settingup.btn_ok.clicked.connect(handleOK)
        new_settingup.btn_close.clicked.connect(handleClose)
        new_settingup.facebank_path.clicked.connect(handleChangeFaceBank)
        new_settingup.btn_default.clicked.connect(handleChangeDefault)
        new_settingup.update_facebank.clicked.connect(handleUpFaceBank)
    # def change_Face_recognize(self, cogf):

    def pressedGendataButton(self):
        if self.generate_dataset_btn.isChecked():
            gen_data_form = TakePic(self)
            def handleOK():
                qm = QMessageBox
                if not gen_data_form.img_name.toPlainText() == '':
                    g_image_name = '%s/%s/%s'%(FACE_BANK, gen_data_form.img_name.toPlainText(), gen_data_form.img_name.toPlainText()) + '_' + datetime.now().strftime('%Y%m%d_%H:%M:%S')+ '.jpg'
                    g_image = gen_data_form.image
                    if os.path.exists('%s/%s'%(FACE_BANK, gen_data_form.img_name.toPlainText())):
                        qm().question(self, '', "%s is exists! Are you continue?"%gen_data_form.img_name.toPlainText(), qm.Yes | qm.No)
                        if qm.Yes:
                            cv2.imwrite(g_image_name, g_image)
                            gen_data_form.close()
                            self.generate_dataset_btn.setText("Generate")
                            self.generate_dataset_btn.setCheckable(False)
                            self.generate_dataset_btn.setCheckable(True)
                    else:
                        os.mkdir('%s/%s'%(FACE_BANK, gen_data_form.img_name.toPlainText()))
                        cv2.imwrite(g_image_name, g_image)
                        gen_data_form.close()
                        self.generate_dataset_btn.setText("Generate")
                        self.generate_dataset_btn.setCheckable(False)
                        self.generate_dataset_btn.setCheckable(True)
                    self.face_recognize.update_facebank()
                else:
                    QMessageBox().about(self, "Validation", "Please enter name person!")


            def handleOCapture():
                if not gen_data_form.allow_capture:
                    gen_data_form.btn_take_pic.setText("Capture")
                else:
                    gen_data_form.btn_take_pic.setText("Again")
                gen_data_form.allow_capture = not gen_data_form.allow_capture

            def handleClose():
                gen_data_form.close()
                self.generate_dataset_btn.setText("Generate")
                self.generate_dataset_btn.setCheckable(False)
                self.generate_dataset_btn.setCheckable(True)
            try:
                self.generate_dataset_btn.setText("Generating")
                gen_data_form.run_video_capture()
                gen_data_form.show()
                
                gen_data_form.btn_ok.clicked.connect(handleOK)
                gen_data_form.btn_close.clicked.connect(handleClose)
                gen_data_form.btn_take_pic.clicked.connect(handleOCapture)
            except:
                msg = QMessageBox()
                self.generate_dataset_btn.setChecked(False)
        else:
            self.generate_dataset_btn.setText("Generate")

    def update_image(self):     # update canvas every time according to time set in the timer.
        if self.recognize_face_btn.isChecked():
            self.ret, self.image = self.capture.read()
            if self.ret:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                image_rc = Image.fromarray(self.image)
                bboxes, faces = self.face_recognize.align_multi(image_rc, thresholds = [0.5, 0.7, 0.8])
                if len(bboxes) != 0:
                    bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                    bboxes = bboxes.astype(int)
                    bboxes = bboxes + [-1,-1,1,1] # personal choice
                    if self.has_targ:   
                        results, score, embs = self.face_recognize.infer(faces, self.targets)
                        for idx, bbox in enumerate(bboxes):
                            draw_box_name(bbox, self.names[results[idx] + 1], self.image)
                    else:
                        for idx, bbox in enumerate(bboxes):
                            draw_box_name(bbox, 'Unknow', self.image)
                self.display()

            else:
                QMessageBox().about(self, "Warring", "Video source not found!")
                self.recognize_face_btn.setText("Recognize")
                self.recognize_face_btn.setCheckable(False)
                self.recognize_face_btn.setCheckable(True)
                self.stop_timer()

        if self.video_recording_btn.isChecked():
            self.recording()

    def save_image(self):       # Save image captured using the save button.
        location = "pictures"
        file_type = ".jpg"
        file_name = self.time()+file_type # a.jpg
        os.makedirs(os.path.join(os.getcwd(),location), exist_ok=True)
        cv2.imwrite(os.path.join(os.getcwd(),location,file_name), self.image)
        QMessageBox().about(self, "Image Saved", "Image saved successfully at "+location+"/"+file_name)

    def save_dataset(self):     # Save images of new dataset generated using generate dataset button.
        location = os.path.join(self.current_path, str(self.dataset_per_subject)+".jpg")
        if self.dataset_per_subject < 1:
            QMessageBox().about(self, "Dataset Generated", "Your response is recorded now you can train the Model \n or Generate New Dataset.")
            self.generate_dataset_btn.setText("Generate Dataset")
            self.generate_dataset_btn.setChecked(False)
            self.stop_timer()
            self.dataset_per_subject = 50 # again setting max datasets

        if self.generate_dataset_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces = self.get_faces()
            self.draw_rectangle(faces)
            if len(faces) is not 1:
                self.draw_text("Only One Person at a time")
            else:
                for (x, y, w, h) in faces:
                    cv2.imwrite(location, self.resize_image(self.get_gray_image()[y:y+h, x:x+w], 92, 112))
                    self.draw_text("/".join(location.split("/")[-3:]), 20, 20+ self.dataset_per_subject)
                    self.dataset_per_subject -= 1
                    self.progress_bar_generate.setValue(100 - self.dataset_per_subject*2 % 100)
        if self.video_recording_btn.isChecked():
            self.recording()
        self.display()

    def display(self):      # Display in the canvas, video feed.
        if self.image is not None:
            pixImage = self.pix_image(self.image)
            self.video_feed.setPixmap(QtGui.QPixmap.fromImage(pixImage))
            self.video_feed.setScaledContents(True)

    def pix_image(self, image): # Converting image from OpenCv to PyQT compatible image.
        qformat = QtGui.QImage.Format_RGB888  # only RGB Image
        if len(image.shape) >= 3:
            r, c, ch = image.shape
        else:
            r, c = image.shape
            qformat = QtGui.QImage.Format_Indexed8
        pixImage = QtGui.QImage(image, c, r, image.strides[0], qformat)
        return pixImage#.rgbSwapped()

    def recognize(self):        # When recognized button is called.
        if self.recognize_face_btn.isChecked():
            self.start_timer()
            self.recognize_face_btn.setText("Stop Recognition")
        else:
            self.recognize_face_btn.setText("Recognize Face")
            self.stop_timer()
            
    def time(self):     # Get current time.
        return datetime.now().strftime("%d-%b-%Y:%I-%M-%S")

    def print_custom_error(self, msg):      # Print custom error message/
        print("="*100)
        print(msg)
        print("="*100)
    def recording(self):        # Record Video when either recognizing or generating.
        if self.ret:
            self.video_output.write(cv2.resize(self.image, (1280,720)))
            
    def save_video(self):       # Saving video.
        if self.video_recording_btn.isChecked() and self.ret:
            self.video_recording_btn.setText("Stop")
            try:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_file_name = self.time()+'.avi'
                path = os.path.join(os.getcwd(),"recordings")
                os.makedirs(path, exist_ok=True)
                self.video_output = cv2.VideoWriter(os.path.join(path,output_file_name),fourcc, 20.0, (1280,720))
            except Exception as e:
                self.print_custom_error("Unable to Record Video Due to")
        else:
            self.video_recording_btn.setText("Record")
            self.video_recording_btn.setChecked(False)
            if self.ret:
                QMessageBox().about(self, "Recording Complete","Video clip successfully recorded into current recording folder")
            else:
                QMessageBox().about(self, "Information", '''Start either datasets generation or recognition First!  ''')

    # Main Menu
    def createMenus(self):
        self.fileMenu = self.menu_bar.addMenu("&File")
    
    def about_info(self):       # Menu Information of info button of application.
        msg_box = QMessageBox()
        msg_box.setText('''
            
        ''')
        msg_box.setInformativeText('''
            
            ''')
        msg_box.setWindowTitle("About AUFR")
        msg_box.exec_()

    def help_info(self):       # Menu Information of help button of application.
        msg_box = QMessageBox()
        msg_box.setText('''
            
        ''')
        msg_box.setInformativeText('''
            

            ''')
        msg_box.setWindowTitle("Help")
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = AUFR()         # Running application loop.
    ui.show()
    sys.exit(app.exec_())       #  Exit application.
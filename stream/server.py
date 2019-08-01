# ----- Imports ----- #
import sys
from flask import (Flask, g, render_template, request, redirect, url_for,
    jsonify, Response)
from threading import Thread
import os
# import video_server.metadata as mdata
from datetime import datetime
import numpy as np
import imagezmq
import imutils
import cv2
import asyncio
import time
from io import BytesIO
import base64
import json
from PIL import Image
import torch
torch.backends.cudnn.benchmark = True
sys.path.insert(0, "..")
from api import face_recognize
from utils.utils import draw_box_name, compare
from utils.config import get_config

app = Flask(__name__)

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())
# ----- Constants ----- #
conf = get_config(net_mode = 'ir_se', threshold = 1.22, detect_id = 1)

class webcam_video_stream:
    def __init__(self, width=None, height=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.width = width
        self.height = height
        self.image_hub = imagezmq.ImageHub()
        self.frame_ori = {}
        self.last_active = {}
        self.active_check_secon = 30
        self.stop_process={}
        self.frame_process={}
        self.lst_thread = []
        self.face_reg = face_recognize(conf)
        self.targets, self.names  = self.face_reg.load_facebank()
    # def Init(self):
    #     self.image_hub = imagezmq.ImageHub()
    def face_verify(self, avtive_key): 
        while 1:
            image_np = self.frame_ori[avtive_key].copy()
            image = Image.fromarray(image_np)
            
            try:
                bboxes, faces = self.face_reg.align_multi(image)
            except:
                # pass
                bboxes = []
                faces = []
            
            if len(bboxes) != 0:
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice   
                
                results, score, embs = self.face_reg.infer(faces, self.targets)
                for idx, bbox in enumerate(bboxes):
                    if False:
                        image_np = draw_box_name(bbox, self.names[results[idx] + 1] + '_{:.2f}__{:.7s}'.format(score[idx], status), image_np)
                    else:
                        image_np = draw_box_name(bbox, self.names[results[idx] +1 ], image_np)
            self.frame_process[avtive_key] = {
                'image': image_np
            }
        # return image_np
    def start(self):
        # start the thread to read frames from the video stream
        self.cam_thread = Thread(target=self.update, args=())
        self.cam_thread.start()
        self.lst_thread.append(self.cam_thread)
        
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        last_active_check = datetime.now()
        new_process = False
        while 1:
            (rpi_name, frame) = self.image_hub.recv_image()
            self.image_hub.send_reply(b'OK')
            # if a device is not in the last active dictionary then it means
            # that its a newly connected device
            if rpi_name not in self.last_active.keys():
                new_process = True
                print("[INFO] receiving data from {}...".format(rpi_name))
                self.stop_process[rpi_name] = False

            self.last_active[rpi_name] = datetime.now()
            self.frame_ori[rpi_name] = frame

            if new_process:      #create one thread 
                setattr(self, 't_%s'%rpi_name, Thread(target=self.face_verify, args=(rpi_name, )))
                getattr(self, 't_%s'%rpi_name).start()
                self.lst_thread.append(getattr(self, 't_%s'%rpi_name))
                new_process = False

            if (datetime.now() - last_active_check).seconds > self.active_check_secon:        # process if 
            # loop over all previously active devices
                for (rpi_name, ts) in list(self.last_active.items()):
                    # remove the RPi from the last active and frame
                    # dictionaries if the device hasn't been active recently
                    if (datetime.now() - ts).seconds > self.active_check_secon:
                        print("[INFO] lost connection to {}".format(rpi_name))
                        self.last_active.pop(rpi_name)
                        self.frame_ori.pop(rpi_name)
                        self.frame_process.pop(rpi_name)
                        t = getattr(self, 't_%s'%rpi_name)
                        t.join()         
                        self.lst_thread.remove(t)
                        self.stop_process[rpi_name] = True

                last_active_check = datetime.now()

    def read(self, rpi_name=None):
        # return the frame most recently read
        if rpi_name is None:
            return None
        return self.frame_ori[rpi_name]




# ----- Functions ----- #
# ----- Routes ----- #

@app.route('/')
def index():
    """Displays the homepage."""
    return render_template('index.html')
@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/realstop', methods=['GET', 'POST'])
def realstop():
    video_server.stop_process['video'] = True
    return render_template('index.html')


@app.route('/camera/<cli_name>')
def camera(cli_name):
    video_server.stop_process[cli_name] = False
    def generate(cli_name):
        while 1:
            cli_frame = video_server.frame_process[cli_name]
            payload = cv2.imencode('.jpg', cli_frame['image'])[1].tobytes()
            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
            if video_server.stop_process[cli_name]:
                break
    return Response(generate(cli_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/Cameras/<lstCliName>')
# def MultiRealPros(lstCliName):
    
#     async def generate(_frame):
#         await asyncio.sleep(0)
#         payload = recognize_webcam(_frame)
#         await asyncio.sleep(0)
#         return payload
#         # yield (b'--frame\r\n'
#         #                            b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')

#     loop = asyncio.get_event_loop()
#     lstCam = []
#     cliActive = video_server.lastActive.keys()
#     for cli_name in lstCliName:
#         if cli_name in cliActive:
#             lstCam.append(cli_name)

#     all_groups = asyncio.gather(*[generate(video_server.read(cli_key)) for cli_key in lstCam])
#     frame_dict = loop.run_until_complete(all_groups)
#     loop.close()
#     return render_template('main.html', frame_dict = frame_dict)

@app.route('/settings')
def settings():
    """Displays the settings page."""
    return render_template('settings.html', locations=locations)

# app.secret_key = 'super secret key'
# app.config['SESSION_TYPE'] = 'filesystem'
video_server =  webcam_video_stream()
video_server.start()
if __name__ == '__main__':

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)

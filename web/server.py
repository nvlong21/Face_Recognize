# ----- Imports ----- #
import sys
from flask import (Flask, g, render_template, request, redirect, url_for,
    jsonify)
from threading import Thread
import os

from .scan import sync
from .db import Database
import video_server.metadata as mdata
from imutils import build_montages
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
from queue import Queue
import torch
from PIL import Image
torch.backends.cudnn.benchmark = True
sys.path.insert(0, "..")
from api import face_recognize
from utils.utils import draw_box_name, compare
from utils.config import get_config
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())


conf = get_config(net_mode = 'mobi', threshold = 1.22, detect_id = 1)
face_recognize = face_recognize(conf)
targets, names  = face_recognize.update_facebank()



class WebcamVideoStream:
    def __init__(self, width=None, height=None):
        # initialize the video camera stream and read the first frame
        # from the stream

        self.width = width
        self.height = height
        self.stopped = False
        self.imageHub = imagezmq.ImageHub()
        self.frameDict = {}
        self.lastActive = {}
        self.lastActiveCheck = datetime.now()
        self.activeCheckSecon = 30

    def Init(self):
        self.imageHub = imagezmq.ImageHub()

    def Start(self):
        # start the thread to read frames from the video stream
        self.camThread = Thread(target=self.Update, args=())
        self.camThread.start()
        return self

    def Update(self):
        # keep looping infinitely until the thread is stopped
        lastActiveCheck = datetime.now()
        while 1:
            (rpiName, frame) = self.imageHub.recv_image()
            self.imageHub.send_reply(b'OK')

            # if a device is not in the last active dictionary then it means
            # that its a newly connected device
            if rpiName not in self.lastActive.keys():
                print("[INFO] receiving data from {}...".format(rpiName))

            self.lastActive[rpiName] = datetime.now()
            self.frameDict[rpiName] = frame

            if (datetime.now() - lastActiveCheck).seconds > self.activeCheckSecon:
            # loop over all previously active devices
                for (rpiName, ts) in list(self.lastActive.items()):
                    # remove the RPi from the last active and frame
                    # dictionaries if the device hasn't been active recently
                    if (datetime.now() - ts).seconds > self.activeCheckSecon:
                        print("[INFO] lost connection to {}".format(rpiName))
                        self.lastActive.pop(rpiName)
                        self.frameDict.pop(rpiName)
                lastActiveCheck = datetime.now()


    def Read(self, rpi_name=None):
        # return the frame most recently read
        if rpi_name is None:
            return None
        return self.frameDict[rpi_name]

    def Stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def recognize_webcam(image_np):
    # image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_np)
    # try:
    bboxes, faces = face_recognize.align_multi(image)
    # except:
    #     # pass
    #     bboxes = []
    #     faces = []

    if len(bboxes) != 0:
        bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
        bboxes = bboxes.astype(int)
        bboxes = bboxes + [-1,-1,1,1] # personal choice   
        
        results, score, embs = face_recognize.infer(faces, targets)
        for idx, bbox in enumerate(bboxes):
            if False:
                image_np = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}__{:.7s}'.format(score[idx], status), image_np)
            else:
                image_np = draw_box_name(bbox, names[results[idx] +1 ], image_np)
    return image_np

# ----- Constants ----- #

# The database schema file.
DB_SCHEMA = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    'schema.sql')

# The database file.
DB_FILE = 'media.db'

# Location of the app media directory.
MEDIA_DIR = 'media'

# Media url.
MEDIA_URL = 'media'

# Possible media types.
MEDIA_TYPES = ['movie', 'show', 'episode']

# Current working directory.
WORKING_DIR = os.getcwd()

# ----- Setup ----- #

# The app object.
app = Flask(__name__)
video_server =  WebcamVideoStream()
video_server.Start()
# Handles database connections and queries.
db = Database(DB_FILE, DB_SCHEMA)

# Creates the media directory.
if not os.path.exists(MEDIA_DIR):
    os.mkdir(MEDIA_DIR)


# ----- Functions ----- #

def _metadata_redirect(media_id, media_type):

    """Parameters for redirect after updating metadata."""

    if media_type == 'movie':

        route = 'movie'
        route_args = {'movie_id': media_id}

    elif media_type == 'show':

        route = 'tv_show'
        route_args = {'show_id': media_id}

    elif media_type == 'episode':

        route = 'episode'
        route_args = {'episode_id': media_id}

    return route, route_args


# ----- Routes ----- #

@app.route('/')
def index():

    """Displays the homepage."""
    return render_template('index.html')

@app.route('/Camera/<cliName>')
def RealPros(cliName):
    cliFrame = video_server.Read(cliName)
    def generate(_frame):
        payload = recognize_webcam(_frame)
        yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
    return Response(generate(cliFrame), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Cameras/<lstCliName>')
async def MultiRealPros(lstCliName):
    
    def generate(_frame):
        payload = recognize_webcam(_frame)
        await asyncio.sleep(0)
        yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')

    loop = asyncio.get_event_loop()
    lstCam = []
    cliActive = video_server.lastActive.keys()
    for cli_name in lstCliName:
        if cli_name in cliActive:
            lstCam.append(cli_name)

    results = asyncio.gather(*[generate(video_server.Read(cli_key)) for cli_key in lstCam])
    frame_dict = loop.run_until_complete(all_groups)
    loop.close()
    return render_template('main.html', frame_dict = frame_dict)

@app.route('/settings')
def settings():

    """Displays the settings page."""

    locations = db.query('SELECT * FROM media_locations')

    return render_template('settings.html', locations=locations)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

if __name__ == '__main__':

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)

import sys
import tempfile
import cv2
import time
import argparse
import datetime
from io import BytesIO
import base64
import json
import numpy as np
from queue import Queue
from threading import Thread
import torch
from flask_wtf.file import FileField
# from importlib import import_module
import os
from flask import Flask, render_template, Response, url_for, session, request, redirect
import numpy as np
from PIL import Image
# from PIL import ImageDraw
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError
torch.backends.cudnn.benchmark = True
sys.path.insert(0, "..")
from api import face_recognize
from utils.utils import draw_box_name, compare
from utils.config import get_config

from sklearn.metrics.pairwise import cosine_similarity
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())
que_name = []
# Helper Functions
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()

class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.src = src
        self.width = width
        self.height = height
        self.stopped = False
        self.flag = 1
        self.count_frame = 0

    def init(self):
        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        (self.grabbed, self.frame) = self.stream.read()

    def start(self):
        # start the thread to read frames from the video stream
        self.camthread = Thread(target=self.update, args=())
        self.camthread.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.count_frame+=1


    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image

def encode_image(image):
  image_buffer = io.BytesIO()
  image.save(image_buffer, format='PNG')
  mime_str = 'data:image/png;base64,'
  imgstr = '{0!s}'.format(base64.b64encode(image_buffer.getvalue()))
  quote_index = imgstr.find("b'")
  end_quote_index = imgstr.find("'", quote_index+2)
  imgstr = imgstr[quote_index+2:end_quote_index]
  imgstr = mime_str + imgstr
  return imgstr

# Webcam feed Helper
def worker(input_q, output_q):
    fps = FPS().start()
    k = 0
    while True:
        fps.update()
        frame = input_q.get()
        # k+=1
        # try:
        # if k%10==0:
        output_q.put(recognize_webcam(frame))
        # else:
        #     output_q.put(frame)
        # except:
        #     pass
    fps.stop()

# detector for web camera
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
            
            is_spoofing = face_recognize.is_spoofing(faces[idx])
            status = "attack"
            if not is_spoofing:
                status = "genuine"
            if len(que_name) > 15:
                que_name.pop()
                que_name.append(names[results[idx] + 1]+ '_{:.7s}'.format(status))
            else:
                que_name.append(names[results[idx] + 1]+ '_{:.7s}'.format(status))
            if False:
                image_np = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}__{:.7s}'.format(score[idx], status), image_np)
            else:
                image_np = draw_box_name(bbox, names[results[idx] +1 ], image_np)
    return image_np


# Image class
class PhotoForm(Form):
  input_photo = FileField(
      'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
      validators=[is_image()])

class VideoForm(Form):
    input_video = FileField()

# recognize function
def recognize(image_path, targets, names):
    frame = cv2.imread(image_path)
    image = Image.fromarray(frame).convert('RGB')
    try:
        bboxes, faces = face_recognize.align_multi(image)
    except:
        bboxes = []
        faces = []
    result = {}
    result['original'] = encode_image(image.copy())

    if len(bboxes) != 0:
        bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
        bboxes = bboxes.astype(int)
        bboxes = bboxes + [-1,-1,1,1] # personal choice   
        results, score, embs = face_recognize.infer(faces, targets)
        
        for idx, bbox in enumerate(bboxes):
            result[names[results[idx] + 1]] = encode_image(faces[idx])
            if False:
                frame = draw_box_name(bbox, names[results[idx]+1] + '_{:.2f}'.format(score[idx]), frame)
            else:
                frame = draw_box_name(bbox, names[results[idx]+1], frame)
    return result


@app.route('/')
def main_display():
    photo_form = PhotoForm(request.form)
    video_form = VideoForm(request.form)
    #return render_template('main.html', photo_form=photo_form, result={})
    return render_template('main.html', photo_form=photo_form, video_form=video_form, result={})

@app.route('/imgproc', methods=['GET', 'POST'])
def imgproc():
    video_form = VideoForm(request.form)
    form = PhotoForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST' and form.validate():
        with tempfile.NamedTemporaryFile() as temp:
            form.input_photo.data.save(temp)
            temp.flush()
            result = recognize(temp.name, targets, names)
        photo_form = PhotoForm(request.form)
        return render_template('main.html',
                               photo_form=photo_form, video_form=video_form, result=result)
    else:
        return redirect(url_for('main_display'))

@app.route('/compare_two_img', methods=['POST'])
def compare_two_img():
    photo_form = PhotoForm(request.form)
    data = request.get_json()
    if data["img_1"].startswith('data:image/jpeg;base64,'):
        img_1 = data["img_1"].replace('data:image/jpeg;base64,', '')
    else:
        img_1 = data["img_1"].replace('data:image/png;base64,', '')
    if data["img_2"].startswith('data:image/jpeg;base64,'):
        img_2 = data["img_2"].replace('data:image/jpeg;base64,', '')
    else:
        img_2 = data["img_2"].replace('data:image/png;base64,', '')

    im = Image.open(BytesIO(base64.b64decode(img_1)))
    im2 = Image.open(BytesIO(base64.b64decode(img_2)))
    im_arr1 = np.array(im)
    im_arr2 = np.array(im2)
    if len(im_arr1.shape) == 2:
        im_arr1 = np.stack([im_arr1] * 3, 2)
        im = Image.fromarray(im_arr1)

    elif im_arr1.shape[2] ==4:
        im_arr1 = im_arr1[:,:,:3]
        im = Image.fromarray(im_arr1)

    if im_arr2.shape[2] == 1:
        im_arr2 = np.stack([im_arr2] * 3, 2)
        im2 = Image.fromarray(im_arr2)

    elif im_arr2.shape[2] == 4:
        im_arr2 = im_arr2[:,:,:3]
        im2 = Image.fromarray(im_arr2)
    print("aaaaaaaaaaaa")

    video_form = VideoForm(request.form)
    features_1, faces_1 = face_recognize.feature_img(im)
    features_2, faces_2 = face_recognize.feature_img(im2)

    if len(faces_1) == 0 or len(faces_1) > 1:
        result = {
            "results": "image 1 should only face!"
        }
    elif len(faces_2) == 0 or len(faces_2) > 1:
        result = {
            "results": "image 2 should only face!"
        }
    else:
        pros, rel = compare(features_1, features_2)

        result = {
                "results": "{0:.2f}".format(pros[0:,0][0]*100)
        }
    return json.dumps(result)

    # return render_template('main.html', photo_form=photo_form, video_form=video_form, result=result)

@app.route('/realproc', methods=['GET', 'POST'])
def realproc():
    video_ids = [0]
    return render_template('realtime.html', video_ids = video_ids)

@app.route('/realstop', methods=['GET', 'POST'])
def realstop():
    photo_form = PhotoForm(request.form)
    video_form = VideoForm(request.form)
    if request.method == 'POST':
        print("In - Stop - POST")
        if request.form['realstop'] == 'Stop Web Cam':
            for vi in lst_video:
                # fps_init.stop()
                vi.stop()
                # vi.update()
            print("Stopped")
    return render_template('main.html', photo_form=photo_form, video_form=video_form)


@app.route('/realpros/<video_id>')
def realpros(video_id):
    video_init = WebcamVideoStream(src = link_video[int(video_id)], width=960, height=720)

    input_q = Queue(5)
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        # t.join()
        t.start()
    video_init.init()
    lst_video.append(video_init)
    video_capture = video_init.start()
    frame = video_capture.read()
    r = [230, 207, 860, 792]
    def generate():
        frame = video_capture.read()
        shape = frame.shape
        x_a = shape[0]-30
        # int(shape[1]/2)-2*len(text)
        if frame is not None:
            frame = frame[r[1]:r[3], r[0]:r[2]]
            count = 0
            while video_capture.grabbed:
                if count % 15 ==0:
                    input_q.put(frame)
                if output_q.empty():
                    pass
                else:
                    data = output_q.get()
                # text = "--".join(que_name)
                # frame = cv2.putText(frame,
                #     text,
                #     ( 10, x_a), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 
                #     1,
                #     (0,255,0),
                #     2,
                #     cv2.LINE_AA)
                    payload = cv2.imencode('.jpg', data)[1].tobytes()
                    # if payload is None:
                    #     payload = cv2.imencode('.jpg', frame)[1].tobytes()
                    yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')

                
                frame = video_capture.read()
                count+=1
            # video_capture.update()
            # fps.update()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/vidproc', methods=['GET', 'POST'])
def vidproc():
    print("In vidproc")
    form = VideoForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST':
        print("vid sub")
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            form.input_video.data.save(temp)
            temp.flush()
            session['vid'] = temp.name
        return render_template('video.html')


@app.route('/vidpros')
def vidpros():
    vid_source = cv2.VideoCapture(session['vid'])
    vid_source.set(cv2.CAP_PROP_POS_MSEC, 25* 1000)
    def generate(names, targets):
        ret, frame = vid_source.read()
        # tensor code
        while ret:
            img_bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_bg)
            try:
                bboxes, faces = face_recognize.align_multi(image)
            except:
                bboxes = []
                faces = []
            if len(bboxes) != 0:
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice   
                results, score, embs = face_recognize.infer(faces, targets)

                for idx, bbox in enumerate(bboxes):
                    if False:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)

            payload = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
            ret, frame = vid_source.read()

    print("Before return")
    return Response(generate(names, targets), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vms')
def vms():
    """Video streaming home page."""
    cameras = [0, 1]
    return render_template('vms.html', cameras = cameras)

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<camera_type>/<device_id>')
def video_feed(camera_type, device_id):
    """Video streaming route. Put this in the src attribute of an img tag."""
    camera_stream = import_module('camera_' + camera_type).Camera
    if camera_type == 'opencv':
        try:
            device = int(device)
        except:
            pass
        camera_stream.set_video_source(link_video)
    return Response(gen(camera_stream(int(device_id))),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

link_video = ['rtsp://admin:a1b2c3d4@@10.0.20.226:554/profile2/media.smp']
# link_video = ['video.mp4']
# client = ObjectDetector()
conf = get_config(net_mode = 'mobi', threshold = 1.22, detect_id = 1)
face_recognize = face_recognize(conf)
targets, names  = face_recognize.load_facebank()
lst_video = []
fps_init = FPS()

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

if __name__ == '__main__':

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)

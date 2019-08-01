import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
import mxnet as mx
import time
thresh = 0.8
scales = [1024, 1024]

count = 1

gpuid = 0
context = mx.cpu()
detector = RetinaFace('./model/mnet.25', 0, -1, 'net3')
stream = cv2.VideoCapture('rtsp://admin:a1b2c3d4@@10.0.20.226:554/profile2/media.smp')
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
_, img = stream.read()
im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
#im_scale = 1.0
#if im_size_min>target_size or im_size_max>max_size:
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

print('im_scale', im_scale)

scales = [im_scale]
while 1:
    ret, img = stream.read()
    img_base = img.copy()
    flip = False
    t = time.time()

    for c in range(count):
        faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
        print(c, faces.shape, landmarks.shape)

    if faces is not None:
        print('find', faces.shape[0], 'faces')
        for i in range(faces.shape[0]):
            #print('score', faces[i][4])
            box = faces[i].astype(np.int)
            #color = (255,0,0)
            color = (0,0,255)
            cv2.rectangle(img_base, (box[0], box[1]), (box[2], box[3]), color, 2)
            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int)
                #print(landmark.shape)
                for l in range(landmark5.shape[0]):
                    color = (0,0,255)
                    if l==0 or l==3:
                        color = (0,255,0)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
    cv2.imshow("a", img_base)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
    print(time.time() - t)
cap.release()


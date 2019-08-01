import sys
import cv2
import os
from PIL import Image
import argparse
import torch
import numpy as np
import time
from datetime import datetime
sys.path.insert(0, "..")
from api import face_recognize
from utils.utils import draw_box_name
from utils.config import get_config


import time
print(torch.cuda.is_available())
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name",default='video.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.25, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    parser.add_argument("-save_unknow", "--save_unknow", help="save unknow person", default=1, type=int)
    link_cam = 'rtsp://admin:a1b2c3d4@@10.0.20.226:554/profile2/media.smp'

    args = parser.parse_args()
    conf = get_config(net_mode = 'mobi', threshold = args.threshold, detect_id = 1)
    face_recognize = face_recognize(conf)
    if args.update:
        targets, names = face_recognize.update_facebank()
        print('facebank updated')
    else:
        targets, names = face_recognize.load_facebank()
        print('facebank loaded')
    if (not isinstance(targets, torch.Tensor)) and face_recognize.use_tensor:
        targets, names = face_recognize.update_facebank()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(args.file_name)

    cap.set(cv2.CAP_PROP_POS_MSEC, args.begin* 1000)

    fps = cap.get(cv2.CAP_PROP_FPS)
    isSuccess, frame = cap.read()
    # r = cv2.selectROI(frame)
    # Crop image

    if args.duration != 0:
        i = 0
    count = 0
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            img_bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_bg)
            # try:
            bboxes, faces = face_recognize.align_multi(image)
            # except:
                # bboxes = []
                # faces = []
            if len(bboxes) != 0:
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice   
                results, score, embs = face_recognize.infer(faces, targets)
                
                for idx, bbox in enumerate(bboxes):
                    # faces[idx].save("%d.jpg"%count)
                    # print(results[idx])
                    count+=1
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            # frame = cv2.resize(frame, (960, 760))
            # video_writer.write(frame)
            cv2.imshow("face_recognize", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  
        else:
            break
        if args.duration != 0:
            i += 1
            if i % 25 == 0:
                print('{} second'.format(i // 25))
            if i > 25 * args.duration:
                break

    cap.release()
    # video_writer.release()

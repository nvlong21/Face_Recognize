import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import sys
from PIL import Image
from RetinaFace_Mx.retinaface import RetinaFace as FaceDetect
import mxnet as mx
from RetinaFace_Mx.align_trans import get_reference_facial_points, warp_and_crop_face
class RetinaFace():
    def __init__(self, gpu_id = -1, thresh = 0.6, scales = [320, 480]):

        self.thresh = thresh
        self.scales = scales
        self.refrence = get_reference_facial_points(default_square= True)

        self.detector = FaceDetect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RetinaFace_Mx/model/mnet.25'), 0, gpu_id, 'net3')
    def align(self, img, limit = None, min_face_size=None, thresholds = None, nms_thresholds=None):
        img = np.array(img)
        im_shape = img.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
        scales_img = [im_scale]
        # img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        for c in range(1):
            boxes, landmarks = self.detector.detect(img, self.thresh, scales=scales_img, do_flip=False)
            
        if len(boxes) ==0:
            return [], []
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]

        faces = []
        for landmark in landmarks:
            warped_face = warp_and_crop_face(img, landmark, self.refrence, crop_size=(112,112))
            return Image.fromarray(warped_face)
        return None
    def align_multi(self, img, limit = None, min_face_size=None, thresholds = None, nms_thresholds=None):

        img = np.array(img)
        im_shape = img.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
        scales_img = [im_scale]
        # img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        for c in range(1):
            boxes, landmarks = self.detector.detect(img, self.thresh, scales=scales_img, do_flip=False)
        if len(boxes) ==0:
            return [], []
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]

        faces = []
        for landmark in landmarks:
            # print(landmark)
            # facial5points = [[landmark[j, 0],landmark[j, 1]] for j in range(5)]
            # print(facial5points)
            warped_face = warp_and_crop_face(img, landmark, self.refrence, crop_size=(112,112))
            faces.append(Image.fromarray(warped_face))

        return boxes, faces
import time
if __name__ == '__main__':
    reti = RetinaFace()
    img = cv2.imread("t6.jpg")

    t = time.time()
    for i in range(10):
	    bboxs, faces = reti.align_multi(img)
	    t2 = time.time()
	    print(t2 -t)
	    t = t2
    i=0
    for face in faces:
        i+=1
        face.save("a%d.jpg"%i)





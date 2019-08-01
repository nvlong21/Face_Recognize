import numpy as np
import cv2
from sklearn.externals import joblib
import os
class Smoofing():
    """docstring for Smoofing"""
    def __init__(self):
        super(Smoofing, self).__init__()
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.clf = joblib.load(os.path.join(current_path,"replay-attack_ycrcb_luv_extraTreesClassifier.pkl"))
        # self.sample_number = 1
        # self.count = 0
        # self.measures = np.zeros(self.sample_number, dtype=np.float)
    def calc_hist(self, img):
        histogram = [0] * 3
        for j in range(3):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            histr *= 255.0 / histr.max()
            histogram[j] = histr
        return np.array(histogram)
    def predict(self, face):
        roi = np.array(face)
        img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

        ycrcb_hist = self.calc_hist(img_ycrcb)
        luv_hist = self.calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))

        prediction = self.clf.predict_proba(feature_vector)
        prob = prediction[0][1]
        if prob > 0.7:
        	return True

        # self.measures[self.count % self.sample_number] = prob
        # point = (x, y-5)
        # if 0 not in self.measures:
        #     text = "True"
        #     if np.mean(self.measures) >= 0.7:
        #         text = "False"
        #         # font = cv2.FONT_HERSHEY_SIMPLEX
        #         # cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
        #         #             thickness=2, lineType=cv2.LINE_AA)
        #     else:
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         # cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9,
        #         #             color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        #     print(text, np.mean(self.measures))
        return False
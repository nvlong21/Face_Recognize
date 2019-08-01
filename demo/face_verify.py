import cv2
from PIL import Image
from pathlib import Path
import torch
from config import get_config
from api import face_recognize
from utils.utils import draw_box_name
import glob
import argparse
from tqdm import tqdm
import pandas as pd
import time
import os

if __name__ == '__main__':
    base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="face recognition")
    parser.add_argument('-image',type=str,help="-image path image")
    # parser.add_argument('-path',type=str,help="-path path folder list image")
    parser.add_argument('-csv',type=str,help="-path path to annotation.csv", default='%s/dataset/annotation.csv'%base_folder)
    parser.add_argument('-path',type=str,help="-path path to image folder", default='%s/dataset/public_test'%base_folder)
    parser.add_argument('-threshold', '--threshold',type=float,help="-threshold threshold", default=1.2)
    parser.add_argument('-use_mtcnn', '--use_mtcnn',type=float,help="using mtcnn", default=1)
    args = parser.parse_args()

    conf = get_config(net_size = 'large', net_mode = 'ir_se',threshold = args.threshold, use_mtcnn = args.use_mtcnn)
    face_recognize = face_recognize(conf)
    targets , names = face_recognize.load_single_face(args.image)

    submiter = [['image','x1','y1','x2','y2','result']]
    sample_df = pd.read_csv(args.csv)
    sample_list = list(sample_df.image)

    for img in tqdm(sample_list):
        temp = [img.split('/')[-1], 0, 0, 0, 0, 0]
        for tp in ['.jpg', '.png', '.jpeg','.img', '.JPG', '.PNG', '.IMG', '.JPEG']:    
            img_path = '%s/%s%s'%(args.path, img, tp)
            if os.path.isfile(img_path):
                break
        image = Image.open(img_path)
        try:
            bboxes, faces = face_recognize.align_multi(image)
        except:
            bboxes = []
            faces = []
        if len(bboxes) > 0:
            bboxes = bboxes[:,:-1] 
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] 
            results, score, _ = face_recognize.infer(faces, targets)
            for id,(re, sc) in enumerate(zip(results, score)):
                if re != -1:
                    temp = [img.split('/')[-1].replace('.png', '.jpg'), bboxes[id][0], bboxes[id][1], bboxes[id][2], bboxes[id][3], 1]
            print(img_path, results)
            break
        submiter.append(temp)
    df = pd.DataFrame.from_records(submiter)
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    df.to_csv("output.csv"%base_folder,index=None)

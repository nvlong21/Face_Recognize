import cv2
from PIL import Image
import torch
from config import get_config
import glob
from tqdm import tqdm
import pandas as pd
import uuid
import subprocess
import zipfile
import os
from api import face_recognize
def download_file_by_url(url, folder_name):
	file_path = folder_name + '/' + url.split('/')[-1]
	command = 'wget %s -P %s'%(url, folder_name)
	subprocess.call(command, shell=True)
	return file_path

def unzip_file(file_zip_path, folder_name):
	path_folder = folder_name + '/unzip'
	zip_ref = zipfile.ZipFile(file_zip_path, 'r')
	zip_ref.extractall(path_folder)
	zip_ref.close()
	return path_folder

def process(data):
	folder_name = str(uuid.uuid1())
	command = 'mkdir %s'%folder_name
	subprocess.call(command, shell=True)
	image_path = download_file_by_url(data['image_url'], folder_name)
	file_zip_path = download_file_by_url(data['file_zip_url'], folder_name)
	path = unzip_file(file_zip_path, folder_name)
	results = process_images(image_path=image_path, path=path)
	command = 'rm -rf %s'%folder_name
	subprocess.call(command, shell=True)
	return results

def process_images(image_path='', path=''):
    
    conf = get_config()
    face_recognize = face_recognize(conf)
    targets, names = face_recognize.load_single_face(image_path)
    submiter = [['image','x1','y1','x2','y2','result']]
    list_file = glob.glob(path + '/*')
    if os.path.isfile(list_file[0]) == False: 
    	path = list_file[0]
    	print(path)
    for img in tqdm(glob.glob(path + '/*')):
        temp = [img.split('/')[-1], 0,0,0,0,0]
        image = Image.open(img)
        try:
            bboxes, faces = face_recognize.align_multi(image)
        except:
            bboxes = []
            faces = []
        if len(bboxes) > 0:
            bboxes = bboxes[:,:-1] 
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] 
            results, score = face_recognize.infer(faces, targets)

            for id,(re, sc) in enumerate(zip(results, score)):
                if re != -1:
                    temp = [img.split('/')[-1], bboxes[id][0], bboxes[id][1], bboxes[id][2], bboxes[id][3], 1]
        submiter.append(temp)
    df = pd.DataFrame.from_records(submiter)
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    df = df.sort_values(by=['result'], ascending=False)
    results = df.to_json(orient='records')
    return results

def process_two_image(data):
    folder_name = str(uuid.uuid1())
    command = 'mkdir %s'%folder_name
    subprocess.call(command, shell=True)
    image_path_origin = download_file_by_url(data['image_url_origin'], folder_name)
    image_path_detection = download_file_by_url(data['image_url_detection'], folder_name)

    from api import face_recognize
    conf = get_config()
    face_recognize = face_recognize(conf)
    face_recognize._raw_load_single_face(image_path_origin)
    targets = face_recognize.embeddings
    image = Image.open(image_path_detection)
    submiter = [['image_url','x1','y1','x2','y2','result']]
    try:
        bboxes, faces = face_recognize.align_multi(image)
    except:
        bboxes = []
        faces = []
    if len(bboxes) > 0:
        bboxes = bboxes[:,:-1] 
        bboxes = bboxes.astype(int)
        bboxes = bboxes + [-1,-1,1,1] 
        results, score = face_recognize.infer(faces, targets)

        for id,(re, sc) in enumerate(zip(results, score)):
            if re != -1:
                temp = {
                	'x1': bboxes[id][0], 
                	'y1': bboxes[id][1], 
                	'x2': bboxes[id][2], 
                	'y2': bboxes[id][3],
                	'result':1
                }
                temp = [data['image_url_detection'], bboxes[id][0], bboxes[id][1], bboxes[id][2], bboxes[id][3], 1]
                submiter.append(temp)
    command = 'rm -rf %s'%folder_name
    subprocess.call(command, shell=True)
    df = pd.DataFrame.from_records(submiter)
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    df = df.sort_values(by=['result'], ascending=False)
    results = df.to_json(orient='records')
    return results
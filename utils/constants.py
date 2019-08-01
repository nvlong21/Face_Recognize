import os
from easydict import EasyDict as edict


WORK_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#for demo
DEMO_PATH = os.path.join(WORK_PATH, 'demo')
APP_PATH = os.path.join(WORK_PATH, 'app')
UTILS_PATH = os.path.join(WORK_PATH, 'utils')
MODEL_PATH = os.path.join(WORK_PATH, 'src')
WEIGHT_DIR = os.path.join(WORK_PATH, 'src', 'weights')
FACE_BANK = os.path.join(MODEL_PATH, 'Face_bank')
list_model = ['wget https://www.dropbox.com/s/akktsgxp0n8cwn2/model_mobilefacenet.pth?dl=0 -O model_mobilefacenet.pth',
'wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O model_ir_se50.pth',
'wget https://www.dropbox.com/s/rxavczg9dlxy3a8/model_ir50.pth?dl=0 -O model_ir50.pth']
WEIGHT_PATH = edict()
WEIGHT_PATH.ir_se = '%s/weights/model_ir_se50.pth'%MODEL_PATH
WEIGHT_PATH.ir = '%s/weights/model_ir50.pth'%MODEL_PATH
WEIGHT_PATH.mobi = '%s/weights/model_mobilefacenet.pth'%MODEL_PATH
URL = edict()
URL.ir_se = list_model[1]
URL.ir = list_model[2]
URL.mobi = list_model[0]
# 
INPUT_SIZE = [112, 112]
THRESHOLD_PERCENT = 0.75
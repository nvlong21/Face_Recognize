from easydict import EasyDict as edict
import torch
import os
from torchvision import transforms as trans
from .constants import *

def get_config(net_mode = 'ir_se', detect_id = 0, threshold = 1.22, use_tensor = False):
    conf = edict()
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.input_size = INPUT_SIZE
    conf.model_path = MODEL_PATH
    conf.facebank_path = FACE_BANK
    conf.face_limit = 5  #when inference, at maximum detect 10 faces in one image, my laptop is slow
    conf.min_face_size = 32.0

    assert net_mode in ['ir_se', 'ir', 'mobi'], 'net_mode should be mobi, ir_se, ir please change in cogfig.py'
    conf.net_mode = net_mode 

    if net_mode == 'ir_se':
        conf.use_mobile_facenet = False
        conf.net_mode = 'ir_se' 
        
    elif net_mode == 'ir':
        conf.use_mobile_facenet = False
        conf.net_mode = 'ir' 
    else:
        conf.use_mobile_facenet = True
    
    conf.threshold = threshold
    conf.use_mtcnn = False
    conf.use_retina = False
    if detect_id==0:
        conf.use_mtcnn = True
    elif detect_id == 1:
        conf.use_retina = True

    conf.use_tensor = use_tensor
    if not  conf.use_tensor:
        conf.threshold = THRESHOLD_PERCENT
    conf.test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    return conf
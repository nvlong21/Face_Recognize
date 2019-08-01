from easydict import EasyDict as edict
# from pathlib import Path
import torch
import os
from torchvision import transforms as trans
from utils.constants import *
list_model = ['wget https://www.dropbox.com/s/akktsgxp0n8cwn2/model_mobilefacenet.pth?dl=0 -O model_mobilefacenet.pth',
'wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O model_ir_se50.pth',
'wget https://www.dropbox.com/s/rxavczg9dlxy3a8/model_ir50.pth?dl=0 -O model_ir50.pth']
def get_config():
    conf = edict()
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.lr = 1e-3
    conf.milestones = [18,30,42]
    conf.momentum = 0.9
    conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
    conf.num_workers = 3
    conf.train_root = "/mnt/01D4A1D481139570/Dataset/Face/casia"
    conf.file_list =  '/mnt/01D4A1D481139570/Dataset/Face/casia_train.txt' 
    conf.batch_size = 4
    conf.lfw_root = '/mnt/01D4A1D481139570/Dataset/Face/data/LFW/lfw_align_112'
    conf.lfw_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/LFW/pairs.txt'
    conf.agedb_root = '/mnt/01D4A1D481139570/Dataset/Face/data/AgeDB-30/agedb30_align_112'
    conf.agedb_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/AgeDB-30/agedb_30_pair.txt'
    conf.cfp_root = '/mnt/01D4A1D481139570/Dataset/Face/data/CFP-FP/CFP_FP_aligned_112'
    conf.cfp_file_list = '/mnt/01D4A1D481139570/Dataset/Face/data/CFP-FP/cfp_fp_pair.txt'
    return conf
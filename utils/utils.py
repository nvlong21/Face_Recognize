from datetime import datetime
from PIL import Image
import numpy as np
import io
from torchvision import transforms as trans
import torch
from src.backbone.model import l2_norm
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.stats import norm
import math
def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

def prepare_facebank(conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = []
    names.append('Unknown')
    for path in tqdm(Path(conf.facebank_path).iterdir()):
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():

                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                        image = np.array(img)
                        if image.shape[2] >3:
                            img = Image.fromarray(image[...,:3])

                    except:
                        continue
                    
                    if img.size != (112, 112):
                        img = mtcnn.align(img)
                    if img is None:
                        continue
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:                        
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, '%s/facebank.pth'%conf.facebank_path)
    np.save('%s/names'%conf.facebank_path, names)
    return embeddings, names
def prepare_facebank_np(conf, model, mtcnn, tta = True):
    model.eval()
    embeddings =  []
    names = []
    names.append('Unknown')
    for path in tqdm(Path(conf.facebank_path).iterdir()):
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():

                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                        image = np.array(img)
                        if image.shape[2] >3:
                            img = Image.fromarray(image[...,:3])

                    except:
                        continue

                    if img.size != (112, 112):
                        img = mtcnn.align(img)
                        print(img)
                    if img is None:
                        continue
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror).data.cpu().numpy())
                        else:                        
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)).data.cpu().numpy())
        if len(embs) == 0:
            continue
        embedding = np.mean(embs,axis=0)
        embeddings.append(embedding[0])
        names.append(path.name)
    embeddings = np.array(embeddings)
    names = np.array(names)
    torch.save(embeddings, '%s/facebank.pth'%conf.facebank_path)
    np.save('%s/names'%conf.facebank_path, names)
    return embeddings, names

def load_facebank(conf):
    embeddings = torch.load('%s/facebank.pth'%conf.facebank_path)
    names = np.load('%s/names.npy'%conf.facebank_path)
    return embeddings, names


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

threshold = 73.18799151798612
mu_0 = 89.6058
sigma_0 = 4.5451
mu_1 = 43.5357
sigma_1 = 8.83

def compare(fn_0, fn_1):
    shape= fn_0.shape[-1]
    x0 = np.divide(fn_0, np.stack([np.linalg.norm(fn_0, axis=-1)]*shape, 1))
    x1 = np.divide(fn_1, np.stack([np.linalg.norm(fn_1, axis=-1)]*shape, 1))
    cosine = np.dot(x0, x1.T)

    theta = np.arccos(cosine)
    theta = theta * 180 / math.pi
    prob = get_prob(theta)
    return prob, theta < threshold


def get_prob(theta):
    prob_0 = norm.pdf(theta, mu_0, sigma_0)
    prob_1 = norm.pdf(theta, mu_1, sigma_1)
    total = prob_0 + prob_1
    return prob_1 / total

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (0,255,0),
                    1,
                    cv2.LINE_AA)
    return frame

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as trans
import math
import time
from utils.utils import load_facebank as _load_facebank
from utils.utils import prepare_facebank, prepare_facebank_np, compare
from utils.constants import WEIGHT_DIR, WEIGHT_PATH, URL
from src.backbone.model import SE_IR, MobileFaceNet, l2_norm
# from smoofing.Smoofing import Smoofing
class face_recognize(object):
    def __init__(self, conf):
        self.setup(conf)
    def setup(self, conf):
        self.conf = conf
        if self.conf.use_mobile_facenet:
            self.model = MobileFaceNet(512).to(conf.device)
        else:
            self.model = SE_IR(50, 0.4, conf.net_mode).to(conf.device)
        self.use_tensor = conf.use_tensor     #If False: su dung numpy dung cho tuong lai khi trien khai qua Product Quantizers cho he thong lon
        self.weight = WEIGHT_PATH[conf.net_mode]
        self.model.eval()
        
        self.threshold = conf.threshold
        self.test_transform = conf.test_transform
        if conf.use_mtcnn:
            from alignment.mtcnn import MTCNN
            self.mtcnn = MTCNN(conf.device, thresholds = [0.5, 0.7, 0.8])
            self.use_mtcnn = True
        else:
            from alignment.detector import Retinaface_Detector
            self.mtcnn = Retinaface_Detector(device = conf.device, thresh = 0.6, scales = [320, 640])
            self.use_mtcnn = False
            self.use_mtcnn = False

        # self.anti_spoofing = Smoofing()

        self.tta = False

        self.limit = conf.face_limit
        self.min_face_size = conf.min_face_size
        self.with_facebank = False
        self.load_state(self.conf.device.type)
        self.load_facebank()
            
    def load_state(self, device='cpu'): 
        if not os.path.isfile(self.weight):
            if not os.path.exists(WEIGHT_DIR):
                os.mkdir(WEIGHT_DIR)
            os.system(URL[self.conf.net_mode])
            os.system('mv %s %s'%(URL[self.conf.net_mode].split(' ')[-1], WEIGHT_DIR))

        if device == 'cpu':        
            self.model.load_state_dict(torch.load(self.weight, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(self.weight))
    def is_spoofing(self, image):
        return self.anti_spoofing.predict(image)     

    def _raw_load_facebank(self):
        self.embeddings = torch.load('%s/facebank.pth'%self.conf.facebank_path)
        self.names = np.load('%s/names.npy'%self.conf.facebank_path)

    def load_single_face(self, image, name='Unknow'):
        embeddings = []
        names = ['Unknown']
        embs = []
        assert not image is None, 'None is not image, please enter image path!'
        try:
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            elif isinstance(image, str):
                assert os.path.isfile(image), 'No such image name: %s'%image
                img = Image.open(image)
            else:
                    img = image   
        except:
            pass
        if img.size != (112, 112):
            img = self.mtcnn.align(img)
        with torch.no_grad():
            if self.tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(self.test_transform(img).to(self.conf.device).unsqueeze(0))
                emb_mirror = self.model(self.test_transform(mirror).to(self.conf.device).unsqueeze(0))
                if self.use_tensor:
                    embs.append(l2_norm(emb + emb_mirror))
                else:
                    embs.append(l2_norm(emb + emb_mirror).data.cpu().numpy())
            else:                        
                embs.append(self.model(self.test_transform(img).to(self.conf.device).unsqueeze(0)))
        if not len(embs) == 0:
            names.append(name)
            names = np.array(names)
            if self.use_tensor:
                embedding = torch.cat(embs).mean(0,keepdim=True)
                embeddings.append(embedding)
                embeddings = torch.cat(embeddings)
            else:
                embedding = np.mean(embs,axis=0)
                embeddings.append(embedding[0])
        return embeddings, names

    def get_facebank(self):
        return self.embeddings, self.names
            
    def update_facebank(self):
        if self.use_tensor:
            self.embeddings, self.names = prepare_facebank(self.conf, self.model, self.mtcnn, self.tta)
        else:
            self.embeddings, self.names = prepare_facebank_np(self.conf, self.model, self.mtcnn, self.tta)
        return self.embeddings, self.names

    def load_facebank(self):
        try:
            self.embeddings, self.names = _load_facebank(self.conf)
        except:
            self.embeddings, self.names =[], []
        assert len(self.embeddings) > 0 or len(self.names) > 0, "Face_bank folder not exists or empty"
        if not self.use_tensor:
            if not isinstance(self.embeddings, np.ndarray):
                self.embeddings = [emb.data.cpu().numpy() for emb in self.embeddings]
        return self.embeddings, self.names

    def align_multi(self, img):
        bboxes, faces = self.mtcnn.align_multi(img, self.limit, self.min_face_size)
        return bboxes, faces

    def align(img):
        face = self.mtcnn.align(img)
        return face
    def feature_img(self, img):
        bboxes, faces = self.align_multi(img)
        embs = []
        for im in faces:
            embs.append(self.model(self.test_transform(im).to(self.conf.device).unsqueeze(0)).data.cpu().numpy()[0])
        return np.array(embs), faces

    def compare(self, feature_1, feature_2):
        return 1.0
        


    def infer(self, faces, target_embs):
        if self.use_tensor:
            min_idx, minimum, source_embs = self.infer_tensor(faces, target_embs)
        else:
            min_idx, minimum, source_embs = self.infer_numpy(faces, target_embs)
        return min_idx, minimum, source_embs

    def infer_tensor(self, faces, target_embs):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if self.tta:
                with torch.no_grad():
                    mirror = trans.functional.hflip(img)
                    emb = self.model(self.test_transform(img).to(self.conf.device).unsqueeze(0))
                    emb_mirror = self.model(self.test_transform(mirror).to(self.conf.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror))
            else:
                with torch.no_grad():                        
                    embs.append(self.model(self.test_transform(img).to(self.conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0).to(self.conf.device)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum, source_embs
    def infer_numpy(self, faces, target_embs):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if self.tta:
                with torch.no_grad():
                    mirror = trans.functional.hflip(img)
                    emb = self.model(self.test_transform(img).to(self.conf.device).unsqueeze(0))
                    emb_mirror = self.model(self.test_transform(mirror).to(self.conf.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror).data.cpu().numpy()[0])
            else:
                with torch.no_grad():                        
                    embs.append(self.model(self.test_transform(img).to(self.conf.device).unsqueeze(0)).data.cpu().numpy()[0])
        source_embs = np.array(embs)
        probs, re = compare(source_embs, target_embs)
        # diff =  source_embs - np.expand_dims(target_embs, 0)
        # dist = np.sum(np.power(diff, 2), axis=2)
        maximum = np.amax(probs, axis=1)

        max_idx = np.argmax(probs, axis=1)
        max_idx[maximum < self.threshold] = -1 # if no match, set idx to -1
        return max_idx, maximum, source_embs



    def take_a_pic(self, name):
        save_path = self.conf.facebank_path/name
        if not save_path.exists():
            save_path.mkdir()
        cap = cv2.VideoCapture(0)
        cap.set(3,800)
        cap.set(4,720)
        while cap.isOpened():
            isSuccess,frame = cap.read()
            if isSuccess:
                frame_text = cv2.putText(frame,
                            'Press t to take a picture,q to quit.....',
                            (10,100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            2,
                            (0,255,0),
                            3,
                            cv2.LINE_AA)
                cv2.imshow("Capture", frame_text)
            if cv2.waitKey(1)&0xFF == ord('t'):
                p =  Image.fromarray(frame[...,::-1])
                try:            
                    warped_face = np.array(self.mtcnn.align(p))[...,::-1]
                    cv2.imwrite(str(save_path/'{}.png'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face)
                except:
                    print('no face captured')
            if cv2.waitKey(1)&0xFF == ord('q'):
                break
        cap.release()
        cv2.destoryAllWindows()


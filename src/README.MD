# Face recognition
## Requirements
Python 3.5+

Commanline:
```
pip3 install -r requirements.txt
```

## Usage:
### Download
```
git clone https://github.com/vanlong96tg/Face_recognize_pytorch
mkdir face_recognize/weights
cd face_recognize/weights
wget https://www.dropbox.com/s/akktsgxp0n8cwn2/model_mobilefacenet.pth?dl=0 -O model_mobilefacenet.pth
wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O model_ir_se50.pth
wget https://www.dropbox.com/s/rxavczg9dlxy3a8/model_ir50.pth?dl=0 -O model_ir50.pth
```
### Python:
Run with default threshold=1.2:
```
python3 face_verify.py -csv {path_sample submit_csv} -path {path_folder_image} -image {path_image}
```
Use model ir_se50 (slower but more accurate): 
```
python3 face_verify.py -csv {path_sample submit_csv}  -path {path_folder_image} -image {path_image}
```
Use model MobileNet change config.py: 
```
change args net_size in get_config with net_size='mobi'
```
Args get_config(mode = 'app', net_size = 'large', net_mode = 'ir_se', use_mtcnn = 1, threshold = 1.25) in config.py:
```
mode: for demo
net_size: 'large' = model SE_IR50, 'mobi' = model MobileNet
net_mode: for net_size='large' value in ['ir_se', 'ir']
use_mtcnn: 1 using mtcnn default: recommend
threshold: distance > threshold => unknow 

```
Use model mtcnn for face detection: 
```
python3 face_verify.py -csv {path_sample submit_csv} -path {path_folder_image} -image {path_image} -user_mtcnn 1
```
Run on video: 
```
python3 infer_on_video.py 
```
[Video](https://www.dropbox.com/s/7g26jvp1j4epo7n/video.mp4?dl=0) and [Face bank](https://www.dropbox.com/s/4pstxap2uozvukc/Face_bank.zip?dl=0).Download video, Face bank and extract in dir.

### Docker:
Install docker
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo groupadd docker
sudo usermod -aG docker $USER
```
Install docker-compose
```
pip3 install docker-compose
```
Run
```
docker-compose up --build -d
```
## Test API with docker
Requirements: url not authenticate

Test with Postman
### Two image detection
**URL:** http://localhost:8084/face_recognition
```
{
	"image_url_origin":"https://www.dropbox.com/s/vm8fvi9xdmjrdmr/PQH_0000.png?dl=0",
	"image_url_detection":"https://www.dropbox.com/s/vm8fvi9xdmjrdmr/PQH_0000.png?dl=0"
}
```
### Multiple image 
Please zip image to file

**URL:** http://localhost:8084/face_recognition_two_image
```
{
	"image_url":"https://www.dropbox.com/s/vm8fvi9xdmjrdmr/PQH_0000.png?dl=0",
	"file_zip_url":"https://www.dropbox.com/s/bf705wgk2n9vog6/test.zip?dl=0"
}
```
### Training:
* Performance

	|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|[CFP_FF](http://www.cfpw.io/paper.pdf)|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|
	|:---:|:---:|:---:|:---:|
	|99.73|99.68|97.32|94.88|

### Acknowledgement 
* This repo is inspired by [InsightFace.MXNet](https://github.com/deepinsight/insightface), [InsightFace.PyTorch](https://github.com/TreB1eN/InsightFace_Pytorch), [ArcFace.PyTorch](https://github.com/ronghuaiyang/arcface-pytorch), [MTCNN.MXNet](https://github.com/pangyupo/mxnet_mtcnn_face_detection) and [PretrainedModels.PyTorch](https://github.com/Cadene/pretrained-models.pytorch).
* Training Datasets [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

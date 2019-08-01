# Face_recognize_system
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
Run demo:
```
cd demo
python infer_on_video.py 
```
Run web_demo:
```
cd web
python app.py
```
Run demo system manager for rasperi:
```
cd stream
python client.py -s ip_adress
python servr.py
```
### Trainning: 
* Performance

	|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|[CFP_FF](http://www.cfpw.io/paper.pdf)|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|
	|:---:|:---:|:---:|:---:|
	|99.73|99.68|97.32|94.88|

### Pretrain:
*Check in folder 'src'. 

### Acknowledgement 
* This repo is inspired by [InsightFace.MXNet](https://github.com/deepinsight/insightface), [InsightFace.PyTorch](https://github.com/TreB1eN/InsightFace_Pytorch), [ArcFace.PyTorch](https://github.com/ronghuaiyang/arcface-pytorch), [MTCNN.MXNet](https://github.com/pangyupo/mxnet_mtcnn_face_detection) and [PretrainedModels.PyTorch](https://github.com/Cadene/pretrained-models.pytorch).
* Training Datasets [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

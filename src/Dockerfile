FROM debian:latest

RUN apt-get -y update && apt-get install -y git python3-pip python3-dev python3-tk procps curl

RUN apt-get install -y libsm6 libxext6

RUN pip3 install Flask pandas torch==0.4.0 numpy==1.14.5 matplotlib==2.1.2 tqdm==4.23.4 mxnet_cu90==1.2.1 scipy==1.0.0 bcolz==1.2.1 easydict==1.7 opencv_python==3.4.0.12 Pillow==5.2.0 mxnet==1.2.1.post1 scikit_learn==0.19.2 tensorboardX==1.2 torchvision==0.2.1

ADD . /face_recognition

WORKDIR face_recognition

RUN apt-get install -y wget

ENV PYTHONPATH=$PYTHONPATH:src
ENV FACE_RECOGNITION_PORT=8084
EXPOSE $FACE_RECOGNITION_PORT

ENTRYPOINT ["python3"]
CMD ["face_recognition.py"]

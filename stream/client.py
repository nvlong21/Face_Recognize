# USAGE
# python client.py --server-ip SERVER_IP

# import the necessary packages
from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
	args["server_ip"]))

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
rpiName = socket.gethostname()
# vs = VideoStream(usePiCamera=False, src = 'video.mp4').start()
vs = cv2.VideoCapture('video.mp4')
vs.set(cv2.CAP_PROP_FPS, 1)
#vs = VideoStream(src=0).start()
rpiName = "video"
while True:
	time.sleep(0.04)
	# read the frame from the camera and send it to the server
	_, frame = vs.read()
	sender.send_image(rpiName, frame)

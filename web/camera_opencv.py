import cv2
from BaseCamera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames(unique_id):
        camera = cv2.VideoCapture(Camera.video_source[unique_id])
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            
            # encode as a jpeg image and return it
            yield img

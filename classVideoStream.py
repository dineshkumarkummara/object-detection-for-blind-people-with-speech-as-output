from threading import Thread
import cv2

class VideoStream:
    def __init__(self, src=0):
        # Initialize the video stream source (default is 0 for webcam)
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Start a thread to update the video stream frames
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                # Stop updating if the thread is flagged to stop
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the latest frame
        return self.frame

    def stop(self):
        # Flag the thread to stop
        self.stopped = True
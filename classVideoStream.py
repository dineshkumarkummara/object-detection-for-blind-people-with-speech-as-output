# Threaded class for performance improvement
from threading import Thread
import cv2
class VideoStream:
	
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False
		
	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
                while True:
                    
                    
                			if self.stopped:
                				return
                			(self.grabbed, self.frame) = self.stream.read()
	
	def read(self):
                # Return the latest frame
		return self.frame
	
	def stop(self):
		self.stopped = True

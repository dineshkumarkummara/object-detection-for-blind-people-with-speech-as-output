import time
import pygame
import cv2
import os
from classVideoStream import VideoStream
from init_model import maybe_download_and_extract
from init_model import create_graph
import tensorflow as tf
import numpy as np
from gtts import gTTS
from init_model import NodeLookup
model_dir=r"C:/Users/dinesh kumar/Documents/inception/qwe"
maybe_download_and_extract(model_dir)
create_graph(model_dir)
#NodeLookup(model_dir)
# Variables declarations
frame_count=0
score=0
start = time.time()
pygame.mixer.init()
pred=0
last=0
human_str=None
font=cv2.FONT_HERSHEY_TRIPLEX
font_color=(255,255,255)

# Init video stream
vs = VideoStream(src=0).start()

# Start tensroflow session
with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        while True:
                frame = vs.read()
                frame_count+=1

                # Only run every 5 frames
                if frame_count%2==0:

                        # Save the image as the fist layer of inception is a DecodeJpeg
                        cv2.imwrite("current_frame.jpg",frame)

                        image_data = tf.gfile.FastGFile("./current_frame.jpg", 'rb').read()
                        predictions=sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})

                        predictions = np.squeeze(predictions)
                        node_lookup = NodeLookup()

                        # change n_pred for more predictions
                        n_pred=1
                        top_k = predictions.argsort()[-n_pred:][::-1]
                        for node_id in top_k:
                                human_str_n = node_lookup.id_to_string(node_id)
                                score = predictions[node_id]
                        if score>.5:
                                # Some manual corrections
                                if human_str_n=="stethoscope":human_str_n="Headphones"
                                if human_str_n=="spatula":human_str_n="fork"
                                if human_str_n=="iPod":human_str_n="iPhone"
                                human_str=human_str_n

                                lst=human_str.split()
                                human_str=" ".join(lst[0:2])
                                human_str_filename=str(lst[0])

                        current= time.time()
                        fps=frame_count/(current-start)

                # Speech module        
                if last>40 and not pygame.mixer.music.get_busy() and human_str==human_str_n:
                        pred+=1
                        name=human_str_filename+".mp3"

                        # Only get from google if we dont have it
                        if not os.path.isfile(name):
                                tts = gTTS(text="I see a "+human_str, lang='en')
                                tts.save(name)

                        last=0
                        pygame.mixer.music.load(name)
                        pygame.mixer.music.play()

                # Show info during some time              
                if last<40 and frame_count>10:
                        # Change the text position depending on your camera resolution
                        cv2.putText(frame,human_str, (20,400),font, 1, font_color)
                        cv2.putText(frame,str(np.round(score,2))+"%",(20,440),font,1,font_color)

                if frame_count>20:
                        fps_text="fps: "+str(np.round(fps,2))
                        cv2.putText(frame, fps_text, (460,460), font, 1, font_color)

                cv2.imshow("Frame", frame)
                last+=1


                # if the 'q' key is pressed, stop the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):break

# cleanup everything
vs.stop()
cv2.destroyAllWindows()     
sess.close()
print("Done")

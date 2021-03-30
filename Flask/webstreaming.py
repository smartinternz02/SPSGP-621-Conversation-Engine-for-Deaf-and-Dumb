import numpy as np
import cv2
import os
from keras.models import load_model
from flask import Flask, render_template, Response
import tensorflow as tf
from gtts import gTTS #to convert text to speech
global graph
global writer
from skimage.transform import resize

graph = tf.get_default_graph()
writer = None

model = load_model('aslpng1.h5')

vals = ['A', 'B','C','D','E','F','G','H','I']

app = Flask(__name__)

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(0) #triggers the local camera

pred=""

#preprocessing the frame captured from camera
def detect(frame):
        img = resize(frame,(64,64,1))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img = img/255.0
        with graph.as_default():
                prediction = model.predict_classes(img)
        print(prediction)
        pred=vals[prediction[0]]
        print(pred)
        return pred


@app.route('/')
def index():
    return render_template('index.html')

def gen():
        while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
            
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break
            
            data = detect(frame)
            # output frame
            text = "It indicates "+data
            cv2.putText(frame, text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 4)
            cv2.imwrite("1.jpg",frame)

            #converts text to speech and plays the audio
            speech = gTTS(text = data, lang = 'en', slow = False)
            speech.save("text.mp3")
            os.system("start text.mp3")
            
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(r"output.avi", fourcc, 25,(frame.shape[1], frame.shape[0]), True)

            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                                bytearray(encodedImage) + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

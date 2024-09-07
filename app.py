from flask import Flask,render_template,Response,request
from deepface import DeepFace
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
# import tensorflow as tf
app=Flask(__name__)
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

my_model = tf.keras.models.load_model('model.h5')
class_labels = ['angry','happy','neutral','sad','surprise']
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)



def face_detection(img,size=0.5):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_roi = face_detect.detectMultiScale(img_gray, 1.3,1)

    for(x,y,w,h) in face_roi:
        x = x - 5
        w = w + 10
        y = y + 7
        h = h + 2
        draw_border(img, (x,y),(x+w,y+h),(0,0,204), 2,15, 10)
        #cv2.imshow('detected_face', img)
        #cv2.waitKey(0)
        img_gray_crop = img_gray[y:y+h,x:x+w]
        img_color_crop = img[y:y+h,x:x+w]
        
        
        final_image = cv2.resize(img_color_crop, (48,48))
        final_image = np.expand_dims(final_image, axis = 0)
        final_image = final_image/255.0
    
        prediction = my_model.predict(final_image)
        label=class_labels[prediction.argmax()]
        cv2.putText(img,label, (x+20, y-40), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (92,79,19),3)

        img_color_crop = cv2.flip(img_color_crop, 1)
    return img

def generate_frames():
    camera=cv2.VideoCapture(cv2.CAP_V4L2)
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            result=face_detection(frame)
            ret,buffer=cv2.imencode('.jpg',result)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/',methods=['GET','POST'])
def index():
    status=0
    if request.method=='POST':
            status=1
    return render_template('index.html',status=status)
@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video')
def video():  
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
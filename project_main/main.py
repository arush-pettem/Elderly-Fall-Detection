from flask import Flask, render_template, Response, jsonify, request
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.patches as patches
from ultralytics import YOLO
import os
import cv2
import torch
from PIL import Image
import time
from twilio.rest import Client
#import keys
import os
from twilio.rest import Client
from flask_cors import CORS, cross_origin

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from ultralytics import YOLO
import pickle
import sklearn
with open('rfc_model.pkl','rb')as file:
    rf_classifier = pickle.load(file)

app = Flask(__name__)
CORS(app)

# Video capture object
cap = cv2.VideoCapture('http://192.168.85.181:8080/video')
cap1=cap
#cap = cv2.VideoCapture(0)
def generate_frames():
    while True:
        success, frame = cap1.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

pref_size = (128, 128)
def predict(frame):
    list_fall=[]
    severe=[]
    bounding_boxes=get_boxes_vid(frame)
    img=frame
    complete_images=[]
    if(bounding_boxes==None):
            return list_fall
    for box in bounding_boxes:
        image_height, image_width, _ = img.shape
        x_min, y_min, x_max, y_max = box[:]
        print(f"Bounding box coordinates: ({x_min:.2f}, {y_min:.2f}), ({x_max:.2f}, {y_max:.2f})")
        complete_images.append(img[int(y_min):int(y_max), int(x_min):int(x_max)])
    print(len(complete_images))
    for cropped_img in complete_images:
        # print(1234)
        cropped_img_resized = cv2.resize(cropped_img, pref_size)
        # plt.imshow(cropped_img_resized)
        # plt.show()
        cropped_img_resized = cropped_img_resized / 255.0
        # print(cropped_img_resized.shape)
        cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)
        predictions = model.predict(cropped_img_resized)
        print(predictions)
        severe.append((predictions[0][0]-0.55)*2)
        if(predictions[0][0]>=0.55):
            list_fall.append(1)
        else:
            list_fall.append(0)
    return list_fall,severe

# def sensor_predict(sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7 , sensor8, sensor9):
def sensor_predict(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    input_data = [[int(x1), int(x2),int(x3), int(x4), int(x5), int(x6), int(x7), int(x8), int(x9)]]
    probabilities = rf_classifier.predict_proba(input_data)[0]
    # print("Class Probabilities:", probabilities)
    predicted_class = np.argmax(probabilities)
    # print("Predicted Class:", predicted_class)
    if (predicted_class==1):
        #send_message()
        return "Fall detected"
    
    else :
        return "No Fall detected"

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/models', methods=['GET', 'POST'])
def models():
    # if request.method == 'POST':
    #     # Get the input data (URL or sensor readings)
    #     input_data = request.form.get('input_data')
    #     if input_data.startswith('http'):
    #         # If the input is a URL, redirect to index.html
    #         caretaker_number = request.form.get('caretaker_number')
    #         return render_template('index.html', video_url=input_data, caretaker_number=caretaker_number)
    #     else:
    #         # If the input is sensor readings, process it here
    #         sensor1 = request.form.get('sensor1')
    #         sensor2 = request.form.get('sensor2')
    #         sensor3 = request.form.get('sensor3')
    #         sensor4 = request.form.get('sensor4')
    #         sensor5 = request.form.get('sensor5')
    #         sensor6 = request.form.get('sensor6')
    #         sensor7 = request.form.get('sensor7')
    #         return render_template('sensorpredict.html')
    return render_template('models.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict_fall():
    success, frame = cap.read()

    if not success:
        success, frame = cap.read()
        return jsonify({"result": "Error: Could not read frame."})
    
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    predictions, severities = predict(frame)
    
    if not predictions:
        result = "No person detected"
    else:
        num_persons = len(predictions)
        falling_persons = [i + 1 for i, pred in enumerate(predictions) if pred == 1]
        falling_severities = [severities[i - 1] for i in falling_persons]
        if falling_persons:
            severity_info = ', '.join([f"Person {person}: Severity {severity:.2f}" for person, severity in zip(falling_persons, falling_severities)])
            result = f"{num_persons} persons detected. Fall detected for {severity_info}."
            message_sent = False
            for number in caretakers_numbers:
                message_sent = send_message(number, result) or message_sent
            if message_sent:
                result += " Emergency message sent to caretakers."
        else:
            result = f"{num_persons} persons detected. No fall detected."

    return jsonify({"result": result})


@app.route('/index')
def index():
    video_url = request.args.get('video_url')
    caretaker_number = request.args.get('caretaker_number')
    return render_template('index.html', video_url=video_url, caretaker_number=caretaker_number)

@app.route('/sensorpredict', methods=['POST'])
def sensorpredict():
    sensor1 = request.form.get('sensor1')
    sensor2 = request.form.get('sensor2')
    sensor3 = request.form.get('sensor3')
    sensor4 = request.form.get('sensor4')
    sensor5 = request.form.get('sensor5')
    sensor6 = request.form.get('sensor6')
    sensor7 = request.form.get('sensor7')
    sensor8 = request.form.get('sensor8')
    sensor9 = request.form.get('sensor9')
    print("hihi",sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8,sensor9,"hihi")
    caretaker_number = request.form.get('caretaker_number')

    result = sensor_predict(sensor1, sensor2, sensor3, sensor4, sensor5, sensor6, sensor7, sensor8, sensor9)
    return render_template('sensorpredict.html', result=result)





    
account_sid = 'ACfba60952e29afc016a72e1add12ee0c8'
auth_token = '99ed836c2c07220ca48d1706db72382e'
twilio_client = Client(account_sid, auth_token)
caretakers_numbers = ['+917386342112']

print(tf.version.VERSION)

model=load_model("pranav_sahil_final.h5")
yolo_model = YOLO("yolov8n.pt")

    # Set the input size using the model's overrides parameter
yolo_model.model.args["imgsz"] = 640
    
def get_boxes_vid(frame):
    img = frame
    image_width, image_height, _ = img.shape
    results = yolo_model(img)
    myboxes = []
    for result in results:
        boxes =result.boxes
        for box in boxes:
            class_id=box.cls
            confidence=box.conf.item()
            x_min, y_min, x_max, y_max = box.xyxy[0]
            if class_id==0 and  confidence > 0.3:
                # print(f"Bounding box coordinates: ({x_min:.2f}, {y_min:.2f}), ({x_max:.2f}, {y_max:.2f})")
                temp = [x_min, y_min, x_max, y_max]
                myboxes.append(temp)
    return myboxes

pref_size = (128, 128)
def pred_vid(frame):
    list_fall=[]
    bounding_boxes=get_boxes_vid(frame)
    img=frame
    complete_images=[]
    if(bounding_boxes==None):
            return list_fall
    for box in bounding_boxes:
        image_height, image_width, _ = img.shape
        x_min, y_min, x_max, y_max = box[:]
        print(f"Bounding box coordinates: ({x_min:.2f}, {y_min:.2f}), ({x_max:.2f}, {y_max:.2f})")
        complete_images.append(img[int(y_min):int(y_max), int(x_min):int(x_max)])
    print(len(complete_images))
    for cropped_img in complete_images:
        print(1234)
        cropped_img_resized = cv2.resize(cropped_img, pref_size)
        # plt.imshow(cropped_img_resized)
        # plt.show()
        cropped_img_resized = cropped_img_resized / 255.0
        # print(cropped_img_resized.shape)
        cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)
        predictions = model.predict(cropped_img_resized)
        print(predictions)
        if(predictions[0][0]>=0.55):
            list_fall.append(1)
        else:
            list_fall.append(0)
    return list_fall
        
def send_message(phone_number, message):
    try:
        message = twilio_client.messages.create(
            body=message,
            from_='+13253137785',
            to=phone_number
            )
        print(f"Message sent to {phone_number}: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending message to {phone_number}: {e}")
        return False

app.run(debug=True)

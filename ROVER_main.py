from flask import Flask, render_template, Response, request 
import cv2 
from picamera2 import Picamera2 # type: ignore 
from ultralytics import YOLO 
import pandas as pd 
import cvzone 
import numpy as np 
import RPi.GPIO as GPIO 
import time 
from datetime import datetime 
import os 
from imutils.video import VideoStream 
from imutils.video import FPS 
import argparse 
import imutils 
 
IMAGE_FOLDER = 'images' 
 
app = Flask(_name_) 
 
mq4 = 18   
flame = 36   
reed = 16    
dht11 = 7 
pan_pin = 13 
tilt_pin = 11 
 
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(pan_pin, GPIO.OUT) 
GPIO.setup(tilt_pin, GPIO.OUT) 
pan_pwm = GPIO.PWM(pan_pin, 50) 
tilt_pwm = GPIO.PWM(tilt_pin, 50)

 
class CameraManager: 
    def _init_(self): 
        self.picam2 = None 
        self.locked = False 
 
    def acquire(self): 
        if not self.locked: 
            self.picam2 = Picamera2() 
            self.picam2.resolution = (1920, 1080) 
            self.picam2.format = "RGB888" 
            self.picam2.configure("preview") 
            self.picam2.start() 
            self.locked = True 
             
         
 
    def release(self): 
        if self.locked: 
            self.picam2.stop() 
            self.locked = False 
 
    def capture_frame(self): 
        if self.locked: 
            frame_bgr = self.picam2.capture_array() 
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) 
            return frame 
        else: 
            return None 
 
camera_manager = CameraManager() 
 
def move_left(): 
    pan_pwm.start(2.5) 
    time.sleep(0.5) 
    pan_pwm.stop() 
     
def move_right(): 
    pan_pwm.start(12.5) 
    time.sleep(0.5) 
    pan_pwm.stop() 
 
 
     
def move_up(): 
    tilt_pwm.start(2.5) 
    time.sleep(0.5) 
    tilt_pwm.stop() 


def move_down(): 
    tilt_pwm.start(12.5) 
    time.sleep(0.5) 
    tilt_pwm.stop() 
 
@app.route('/left') 
def left(): 
    move_left() 
    return 'Left' 
 
@app.route('/right') 
def right(): 
    move_right() 
    return 'Right' 
 
@app.route('/up') 
def up(): 
    move_up() 
    return 'Up' 
 
@app.route('/down') 
def down(): 
    move_down() 
    return 'Down' 
 
@app.route('/') 
def index(): 
    return render_template('home.html') 
 
@app.route('/index') 
def index_page(): 
    return render_template('index.html') 
 
 
 
#Video Streaming 
def generate_frames(): 
    camera_manager.acquire() 
    while True: 
        frame = camera_manager.capture_frame() 
        if frame is not None: 
             
            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes() 
 
            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

 
 
@app.route('/video_feed') 
def video_feed(): 
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace;boundary=frame') 
 
#Object Detection 
def object_detection(): 
    camera_manager.acquire() 
    model = YOLO('yolov5nu.pt') 
 
    with open("coco.txt", "r") as my_file: 
        data = my_file.read() 
 
    class_list = data.split("\n") 
    count = 0 
 
    while True: 
        frame = camera_manager.capture_frame() 
        if frame is not None: 
            count += 1 
 
            if count % 3 != 0: 
                continue 
    
 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            results = model.predict(frame_rgb) 
            a = results[0].boxes.data 
            px = pd.DataFrame(a).astype("float") 
 
            for index, row in px.iterrows(): 
                x1 = int(row[0]) 
                y1 = int(row[1]) 
                x2 = int(row[2]) 
                y2 = int(row[3]) 
                d = int(row[5]) 
                c = class_list[d] 
                 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1) 
             
            # Convert frame to JPEG 
            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes() 
 
            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


             
@app.route('/object_detection') 
def object_detection_feed(): 
    return Response(object_detection(), mimetype='multipart/x-mixed-replace;boundary=frame') 
   
 
#Motion Detections             
def motion_detection(): 
    camera_manager.acquire() 
    prev_frame = camera_manager.capture_frame()   
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) 
 
    while True: 
        frame = camera_manager.capture_frame() 
        if frame is not None: 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
 
 
            #Absolute difference between consecutive frames 
            frame_diff = cv2.absdiff(prev_gray, gray) 
            _, frame_diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY) 
 
            contours, _ = cv2.findContours(frame_diff_thresh, cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE) 
 
            motion_detected = False 
            for contour in contours: 
                if cv2.contourArea(contour) > 1000: 
                    motion_detected = True 
                    break 
 
            if motion_detected: 
                cv2.putText(frame, 'Motion Detected!', (10, 30), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
 
            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes() 
 
            yield (b'--frame\r\n' 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
 
            prev_gray = gray.copy() 
 
@app.route('/motion_detection') 
def motion_detection_feed(): 
    return Response(motion_detection(), mimetype='multipart/x-mixed-replace;boundary=frame')


#Environmental Analysis 
 
def read_methane_sensor(mq4): 
    GPIO.setup(mq4, GPIO.IN) 
    analog_value = GPIO.input(mq4) 
    return analog_value 
 
def read_flame_sensor(flame): 
    GPIO.setup(flame, GPIO.IN) 
    flame_value = GPIO.input(flame) 
    return flame_value 
 
 
def read_reed_sensor(reed): 
    GPIO.setup(reed, GPIO.IN) 
    reed_state = GPIO.input(reed) 
    return reed_state 
 
def read_dht11_sensor(dht11): 
    GPIO.setup(dht11. GPIO.IN) 
    dht11_state = GPIO.input(dht11) 
    return dht11_state 
 
 
 
@app.route('/env_analysis') 
def env_analysis(): 
    methane_value = read_methane_sensor(mq4) 
    flame_value = read_flame_sensor(flame) 
    reed_state = read_reed_sensor(reed) 
    temperature_value = read_dht11_sensor(dht11.temperature) 
    humidity_value = read_dht11_sensor(dht11.humidity) 
 
    methane_gas_level = "No harmful methane gas around" if methane_value < 300 else "Extreme harmful methane gas in the area" 
    flame_detection = "No Flame detected.." if flame_value == GPIO.HIGH else "Flame detected !!!" 
    magnetic_field = "Strong magnetic field around" if reed_state == GPIO.LOW else "No strong magnetic field around" 
    temperature = "Optimal temperature" if temperature_value > 20 and temperature_value < 45 else "extreme temperatures" 
    humidity = "Optimal humidity range" if humidity_value > 40 and humidity_value < 60 else "extreme moisture content" 
 
    return render_template('env_analysis.html', methane_gas_level=methane_gas_level,flame_detection=flame_detection, magnetic_field=magnetic_field, temperature=temperature, humidity=humidity)

#Image Capturing 
@app.route('/capture_image') 
def capture_image(): 
    camera_manager.acquire() 
    frame = camera_manager.capture_frame() 
     
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    filename = f"image_{timestamp}.jpg" 
    if frame is not None: 
        image_path = os.path.join(IMAGE_FOLDER, filename) 
        cv2.imwrite(image_path, frame) 
        return f"Image captured and saved as {image_path}" 
    else: 
        return "Failed to capture image" 
     
 
@app.route('/capturing_image_feed') 
def capturing_image_feed(): 
    return render_template('capturing_image.html') 
 
 
if _name_ == '_main_': 
     
    GPIO.setmode(GPIO.BOARD) 
    app.run(host='0.0.0.0', debug=True) 
    GPIO.cleanup()
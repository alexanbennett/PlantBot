import numpy as np
import cv2
from keras.models import load_model
from picamera import PiCamera
from picamera.array import PiRGBArray
import RPi.GPIO as GPIO
import time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock

# Load ensemble models to classify plant
models = [
    load_model('EnsembleCNN/model_1.h5'),
    load_model('EnsembleCNN/model_2.h5'),
    load_model('EnsembleCNN/model_3.h5')
]

# GPIO setup for soil moisture sensor and relay
MOISTURE_SENSOR_PIN = 17  
RELAY_PIN = 25  
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOISTURE_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  
GPIO.setup(RELAY_PIN, GPIO.OUT)

# Initialize camera
camera = PiCamera()
camera.resolution = (640, 480)


SOIL_MOISTURE_THRESHOLD = 480  
PUMP_DURATION = 3  # Duration to run the water pump in seconds
PREDICTION_INTERVAL = 2  # Interval for taking image and making prediction in seconds, this is done to allow time for the raspberry pi to process these images,
#CNN classification takes 1 second.
SENSOR_NOT_CONNECTED_THRESHOLD = 100 

def capture_image():
    rawCapture = PiRGBArray(camera)
    camera.capture(rawCapture, format='bgr')
    image = rawCapture.array
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    return np.expand_dims(image / 255.0, axis=0)

def ensemble_predict(image):
    predictions = np.mean([model.predict(image) for model in models], axis=0)
    return np.argmax(predictions)

def check_soil_moisture():
    sensor_value = GPIO.input(MOISTURE_SENSOR_PIN)
    if sensor_value < SENSOR_NOT_CONNECTED_THRESHOLD:
        return 'not_connected'
    return sensor_value < SOIL_MOISTURE_THRESHOLD

def activate_pump():
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    time.sleep(PUMP_DURATION)
    GPIO.output(RELAY_PIN, GPIO.LOW)

class PlantBot(App):
    def build(self):
        Window.clearcolor = (0.9, 1, 1, 1)  
        self.root = BoxLayout(orientation='vertical')
        self.logo = Image(source='background/logo.png', size_hint=(1, 0.2))
        self.root.add_widget(self.logo)

        hbox = BoxLayout(orientation='horizontal', size_hint=(1, 0.8))
        self.water_level = Image(source='waterstatus/sensornotconnected.png', size_hint=(0.5, 1))
        hbox.add_widget(self.water_level)

        self.plant_stage = Image(source='plantstatus/germination.png', size_hint=(0.5, 1))
        hbox.add_widget(self.plant_stage)
        self.root.add_widget(hbox)

        Clock.schedule_interval(self.update_status, PREDICTION_INTERVAL)
        return self.root

    def update_status(self, dt):
        #This section updates the UI, it changes the UI to show the plant status
        image = capture_image()
        preprocessed_image = preprocess_image(image)
        stage_index = ensemble_predict(preprocessed_image)
        stages = ['germination', 'growing', 'harvest']
        current_stage = stages[stage_index]

        self.plant_stage.source = f'plantstatus/{current_stage}.png'

        soil_moisture_status = check_soil_moisture()
        if soil_moisture_status == 'not_connected':
            self.water_level.source = 'waterstatus/sensornotconnected.png'
        elif soil_moisture_status:
            self.water_level.source = 'waterstatus/needs_water.png'
            activate_pump()
        else:
            self.water_level.source = 'waterstatus/hydrated.png'

if __name__ == '__main__':
    PlantBot().run()

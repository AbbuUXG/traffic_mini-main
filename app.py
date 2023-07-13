from flask import Flask, render_template, Response
import cv2
import numpy as np
import pickle

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load the trained model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Class names for display
class_names = {
   
    0: 'Speed Limit 20 km/h',
    1: 'Speed Limit 30 km/h',
    2: 'Speed Limit 50 km/h',
    3: 'Speed Limit 60 km/h',
    4: 'Speed Limit 70 km/h',
    5: 'Speed Limit 80 km/h',
    6: 'End of Speed Limit 80 km/h',
    7: 'Speed Limit 100 km/h',
    8: 'Speed Limit 120 km/h',
    9: 'No Passing',
    10: 'No Passing for Vehicles over 3.5 metric tons',
    11: 'Right-of-Way at the Next Intersection',
    12: 'Priority Road',
    13: 'Yield',
    14: 'Stop',
    15: 'No Vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No Entry',
    18: 'General Caution',
    19: 'Dangerous Curve to the Left',
    20: 'Dangerous Curve to the Right',
    21: 'Double Curve',
    22: 'Bumpy Road',
    23: 'Slippery Road',
    24: 'Road Narrows on the Right',
    25: 'Road Work',
    26: 'Traffic Signals',
    27: 'Pedestrians',
    28: 'Children Crossing',
    29: 'Bicycles Crossing',
    30: 'Beware of Ice/Snow',
    31: 'Wild Animals Crossing',
    32: 'End of All Speed and Passing Limits',
    33: 'Turn Right Ahead',
    34: 'Turn Left Ahead',
    35: 'Ahead Only',
    36: 'Go Straight or Right',
    37: 'Go Straight or Left',
    38: 'Keep Right',
    39: 'Keep Left',
    40: 'Roundabout Mandatory',
    41: 'End of No Passing',
}


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

def generate_frames():
    while True:
        # Read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess image
            img = cv2.resize(frame, (32, 32))
            img = preprocessing(img)

            # Reshape and predict image
            img = img.reshape(1, 32, 32, 1)
            predictions = model.predict(img)
            classIndex = np.argmax(predictions)
            probabilityValue = np.amax(predictions)

            # Display class and probability
            class_name = class_names.get(classIndex, 'Unknown')
            cv2.putText(frame, "CLASS: " + class_name, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

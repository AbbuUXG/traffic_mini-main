import numpy as np
import cv2
import pickle

# Camera and model setup
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

# Video capture setup
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load the trained model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Class names for display
class_names = {
    0: 'Speed Limit 20 km/h',
    1: 'Speed Limit 30 km/h',
    # Add more class names here
}

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

while True:
    # Read image from camera
    success, imgOriginal = cap.read()

    # Preprocess image
    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)

    # Reshape and predict image
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)

    # Display class and probability
    class_name = class_names.get(classIndex, 'Unknown')
    cv2.putText(imgOriginal, "CLASS: " + class_name, (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

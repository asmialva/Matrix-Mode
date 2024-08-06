import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from PIL import Image
import PIL.ImageTk
import tkinter as tk
from keras.layers import DepthwiseConv2D

# Define a custom DepthwiseConv2D layer to handle the 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

# Load the model with the custom object
try:
    model = load_model("best_model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video stream or file.")
    exit()

# Create a Tkinter window
window = tk.Tk()
window.title("Facial Emotion Analysis")
window.geometry("1000x700")
label = tk.Label(window)
label.pack()

def show_frame():
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        return

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from image
        roi_gray = cv2.resize(roi_gray, (224, 224))

        # Convert grayscale image to RGB
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

        img_pixels = image.img_to_array(roi_rgb)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))

    # Convert the image to PhotoImage
    im = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    imgtk = PIL.ImageTk.PhotoImage(image=im)

    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Call show_frame again after 10 milliseconds
    window.after(10, show_frame)

show_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model for PET detection (You need to train this separately)
model = load_model('pet_bottle_classifier.h5')  # Load your own trained model

# Preprocessing function to prepare the image for classification
def preprocess_image(img):
    # Resize the image to the size the model expects
    img = cv2.resize(img, (128, 128))  # Assuming your model expects 128x128 images
    # Normalize the image data to 0-1 range
    img = img / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is captured correctly
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the captured image
    preprocessed_img = preprocess_image(frame)

    # Make prediction using the model
    prediction = model.predict(preprocessed_img)

    # Assuming the model outputs a probability where 1 = PET and 0 = non-PET
    if prediction > 0.5:
        print(1)  # PET detected
    else:
        print(0)  # Not PET

    # Display the frame
    cv2.imshow('Bottle Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
cap.release()
cv2.destroyAllWindows()

import streamlit as st
import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

# Load the necessary models and encoders
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
model = pickle.load(open("svm_model_final.pkl", 'rb'))
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        text-align: center;
        color: #4CAF50;
    }
    .result {
        font-size: 24px;
        text-align: center;
        color: #FF5722;
    }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown('<div class="title">Face Recognition App</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)
    
    # Process the image
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    if len(faces) == 0:
        st.markdown('<div class="result">No face detected!</div>', unsafe_allow_html=True)
    else:
        for (x, y, w, h) in faces:
            face_img = rgb_img[y:y+h, x:x+w]
            face_img = cv.resize(face_img, (160, 160))  # Resize to 160x160
            face_img = np.expand_dims(face_img, axis=0)
            ypred = facenet.embeddings(face_img)
            face_name = model.predict(ypred)
            final_name = encoder.inverse_transform(face_name)[0]

            # Display the result
            st.markdown(f'<div class="result">Detected: {final_name}</div>', unsafe_allow_html=True)

            # Optionally, display the image with the detected face
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            st.image(img, channels="BGR", caption="Detected Face", use_container_width=True)

# Run the app with: streamlit run your_script.py
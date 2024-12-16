import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Utility Functions
def preprocess_image(img, target_size):
    """Preprocess the image to match the model's input requirements."""
    img = img.resize(target_size)
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def malaria_predict(img, model):
    """Predict malaria infection."""
    img_array = preprocess_image(img, (150, 150))
    prediction = model.predict(img_array)
    predicted_class = (prediction[0][0] > 0.5).astype("int32")
    label = 'Uninfected' if predicted_class == 1 else 'Infected'
    confidence = prediction[0][0]
    return label, confidence

def pneumonia_predict(img, model):
    """Predict pneumonia from chest X-ray."""
    img = img.convert('L')  # Convert to grayscale
    img_array = preprocess_image(img, (36, 36))
    prediction = np.argmax(model.predict(img_array)[0])
    label = 'Pneumonia Detected' if prediction == 1 else 'Normal'
    return label

def predict(values, model_path):
    """Predict using tabular data-based models."""
    model = pickle.load(open(model_path, 'rb'))
    values = np.asarray(values).reshape(1, -1)
    return model.predict(values)[0]

# Streamlit App
st.title("Medical Prediction Application")

# Sidebar Navigation
menu = ["Home","Liver","Pneumonia"]
choice = st.sidebar.selectbox("Navigation", menu)

# Pages
if choice == "Home":
    st.subheader("Welcome to the Medical Prediction Application")
    st.write("Select a disease from the sidebar to make predictions.")

elif choice == "Diabetes":
    st.subheader("Diabetes Prediction")
    with st.form("diabetes_form"):
        values = [st.number_input(f"Enter Value {i+1}", value=0.0) for i in range(8)]
        submitted = st.form_submit_button("Predict")
        if submitted:
            pred = predict(values, 'models/diabetes.pkl')
            st.success(f"Prediction: {'Diabetic' if pred == 1 else 'Non-Diabetic'}")

elif choice == "Breast Cancer":
    st.subheader("Breast Cancer Prediction")
    with st.form("cancer_form"):
        values = [st.number_input(f"Enter Value {i+1}", value=0.0) for i in range(26)]
        submitted = st.form_submit_button("Predict")
        if submitted:
            pred = predict(values, 'models/breast_cancer.pkl')
            st.success(f"Prediction: {'Malignant' if pred == 1 else 'Benign'}")

elif choice == "Liver":
    st.subheader("Liver Disease Prediction")
    with st.form("liver_form"):
        values = [st.number_input(f"Enter Value {i+1}", value=0.0) for i in range(10)]
        submitted = st.form_submit_button("Predict")
        if submitted:
            pred = predict(values, 'models/liver.pkl')
            st.success(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")

elif choice == "Malaria":
    st.subheader("Malaria Detection")
    uploaded_file = st.file_uploader("Upload a Cell Image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        try:
            model = load_model("models/malaria.h5")
            label, confidence = malaria_predict(img, model)
            st.success(f"Prediction: {label} with confidence {confidence:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Pneumonia":
    st.subheader("Pneumonia Detection")
    uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "png","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        st.image(img, caption="Uploaded Image", use_container_width=True)
        img = img.resize((36, 36))
        img_array = np.asarray(img).reshape((1, 36, 36, 1)) / 255.0
        model = load_model("models/pneumonia.h5")
        pred = np.argmax(model.predict(img_array)[0])
        st.success(f"Prediction: {'Pneumonia Detected' if pred == 1 else 'Normal'}")


# Sidebar Info
st.sidebar.info("Use this application for educational purposes only. Always consult a medical professional for accurate diagnosis.")

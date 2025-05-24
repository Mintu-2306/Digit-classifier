import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import joblib

# Load the trained model
log_reg = joblib.load("models/logistic_regression_mnist_model.pkl")

st.title("Minimal Digit Classifier Test")

st.write("Draw a digit below and click Classify.")

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert canvas image to grayscale PIL Image
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
    st.image(img, caption="Your drawing (grayscale)")

    if st.button("Classify"):
        # Resize to 28x28 (MNIST size)
        img_resized = img.resize((28, 28))

        # Convert to numpy array
        img_array = np.array(img_resized)

        # Invert colors (MNIST digits are white on black background)
        img_array = 255 - img_array

        # Normalize pixel values (0-1)
        img_array = img_array / 255.0

        # Flatten image to 1D array
        img_flat = img_array.flatten().reshape(1, -1)

        # Predict digit
        prediction = log_reg.predict(img_flat)[0]

        st.write(f"Predicted digit: **{prediction}**")

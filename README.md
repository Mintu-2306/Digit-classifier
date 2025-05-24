# MNIST Digit Classifier

A simple web application that allows users to draw handwritten digits and classifies them using a Logistic Regression model trained on the MNIST dataset.

---

## 🚀 Features

- **Interactive drawing canvas** with adjustable stroke width
- **Preprocessing** of input images to match MNIST format (28x28 grayscale)
- **Real-time digit classification** using a trained Logistic Regression model
- Lightweight and fast — suitable for demos and learning purposes

---

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Mintu-2306/Digit-classifier.git
   cd Digit-classifier
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
## ▶️ Usage
Start the Streamlit app:
```bash
streamlit run app.py
```
- The app will open in your default web browser.

- Draw a digit (0-9) on the canvas.

- Click **Classify** to see the predicted digit.
## 🗂️ Project Structure
``` File/Folder	Description
app.py	Streamlit web app frontend
models/logistic_regression_mnist_model.pkl	Trained Logistic Regression model
requirements.txt	Python dependencies
```
## 📊 Model Details
- Dataset: MNIST handwritten digits (70,000 images)

- Model: Logistic Regression from Scikit-learn

- Preprocessing: Pixel normalization (0 to 1), resizing input images to 28x28 grayscale

## 📞 Contact
Feel free to open an issue or reach out via GitHub if you have questions or suggestions.

##📝 License
This project is open source and free to use.




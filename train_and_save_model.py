from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib
import os

# Step 1: Load the dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.uint8)

# Optional: Use a smaller subset for even faster training
# X = X[:20000]
# y = y[:20000]

# Step 2: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Build the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),              # Feature scaling
    ('pca', PCA(n_components=100)),            # Dimensionality reduction
    ('log_reg', LogisticRegression(            # Logistic Regression model
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        multi_class='multinomial',
        n_jobs=-1
    ))
])

# Step 4: Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)

# Step 5: Evaluate the model
print("Evaluating the model...")
y_pred = pipeline.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/optimized_logistic_regression_mnist.pkl")
print("Model saved to 'models/optimized_logistic_regression_mnist.pkl'")

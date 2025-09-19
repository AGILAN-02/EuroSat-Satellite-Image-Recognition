import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ==============================================================================
# --- USER CONFIGURATION ---
# ==============================================================================
# IMPORTANT: Update this path to the new image you want to classify
new_image_path = r"D:\Projects\computer vision\version 2.4\Eurosat Dataset\examples\AnnualCrop_119.jpg"
# IMPORTANT: Update this path to the root directory of your dataset"D:\Projects\computer vision\version 2.4\Eurosat Dataset\examples\Residential_330.jpg"
# We need this to get the class labels in the correct order.
DATA_DIR = "D:\Projects\computer vision\EuroSat Dataset"

# Model file to load
model_filename = "eurosat_best_model.keras"  # You can also use "best_model.keras"

# Model's expected image size
IMG_SIZE = 128

# ==============================================================================
# --- PREDICTION SCRIPT ---
# ==============================================================================

# Get the script's directory to correctly locate the model file
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, model_filename)

# Load the saved model
try:
    loaded_model = tf.keras.models.load_model(model_path)
    print(f"✅ Successfully loaded model from {model_path}")
except OSError as e:
    print(f"❌ Error loading model: {e}")
    print(f"Please ensure '{model_filename}' is in the same directory as 'predict.py'.")
    exit()

# Get the class labels from the dataset directory structure
try:
    # This assumes your classes are the subdirectories of DATA_DIR
    class_labels = sorted(os.listdir(DATA_DIR))
    print(f"Class labels detected: {class_labels}")
except FileNotFoundError:
    print(f"❌ Error: DATA_DIR path not found. Please verify the path.")
    exit()

# Load and preprocess the new image
try:
    img = load_img(new_image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
except FileNotFoundError:
    print(f"❌ Error: The image file at {new_image_path} was not found.")
    exit()

# Make the prediction
predictions = loaded_model.predict(img_array)

# Interpret the prediction
predicted_class_index = np.argmax(predictions, axis=1)[0]
predicted_class_name = class_labels[predicted_class_index]
confidence = np.max(predictions) * 100

print("\nPrediction Results:")
print(f"Image: '{os.path.basename(new_image_path)}'")
print(f"Predicted class: {predicted_class_name}")
print(f"Confidence: {confidence:.2f}%")
print("---------------------------")
print("All Class Probabilities:")
for i, label in enumerate(class_labels):
    print(f"  {label}: {predictions[0][i]*100:.2f}%")
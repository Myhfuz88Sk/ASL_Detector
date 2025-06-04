from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'

# Load model
model = load_model('model/asl_detection_model.h5')

# Load class indices safely using JSON
with open('model/class_indices.txt', 'r') as f:
    class_indices = json.load(f)

# Reverse the dictionary to map index to class name
index_to_class = {v: k for k, v in class_indices.items()}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Load and preprocess image using OpenCV
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        img_array = np.expand_dims(img, axis=0)

        # Confirm shape compatibility
        if model.input_shape[1:] != img_array.shape[1:]:
            return jsonify({'error': f"Image shape mismatch. Model expects {model.input_shape[1:]}, got {img_array.shape[1:]}"})

        # Predict
        predictions = model.predict(img_array)
        print("Prediction probabilities:", predictions[0])  # Debug

        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = index_to_class.get(predicted_index, "Unknown")
        confidence = float(np.max(predictions[0])) * 100

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': 'Prediction failed. Details: ' + str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

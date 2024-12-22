from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Grassy_Terrain', 'Marshy_Terrain', 'Other_Image', 'Rocky_Terrain', 'Sandy_Terrain']

# Load the model
model = None

def load_model():
    global model
    model = tf.keras.models.load_model('best_model.keras')

def preprocess_image(image):
    # Convert PIL Image to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Resize image
    img_array = tf.image.resize(img_array, IMG_SIZE)
    # Normalize
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_terrain(image):
    # Preprocess the image
    img_array = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {
            'class': CLASS_NAMES[idx],
            'confidence': float(predictions[0][idx])
        }
        for idx in top_3_idx
    ]
    
    return predicted_class, confidence, top_3_predictions

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image file was sent
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['image']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read and verify image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        predicted_class, confidence, top_3_predictions = predict_terrain(image)
        
        # Return results
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load the model when starting the server
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")
    
    # Start Flask server
    app.run(debug=True, port=5000)

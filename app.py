from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once
MODEL_PATH = 'model.h5'
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

def preprocess_image(image, target_size=(100, 100)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr.reshape((1, target_size[0], target_size[1], 3))
    return arr


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        # Load model if not loaded
        if model is None:
            if not os.path.exists(MODEL_PATH):
                return render_template('index.html', result='Error: Model file not found (model.h5). Please train and save the model first.'), 500
            model = load_model(MODEL_PATH)
            print("Model loaded successfully")

        # Check if image is in request
        if 'image' not in request.files:
            print("No image in request")
            return redirect(url_for('index'))

        file = request.files['image']
        if file.filename == '':
            print("Empty filename")
            return redirect(url_for('index'))

        # Process image
        print(f"Processing image: {file.filename}")
        img = Image.open(file.stream)
        x = preprocess_image(img, target_size=(100, 100))
        
        # Get prediction
        proba = model.predict(x)[0][0]
        
        # Use confidence margin approach:
        # When model is trained only on dogs/cats, it will output:
        # - Close to 0 for dogs
        # - Close to 1 for cats
        # - Close to 0.5 for images it hasn't seen before (uncertain)
        
        confidence_margin = abs(proba - 0.5)  # Distance from uncertain middle point
        
        # Only classify if model is confident (far from 0.5)
        # Require at least 0.3 margin (prob < 0.2 or prob > 0.8)
        if confidence_margin > 0.3:
            if proba > 0.5:
                label = 'cat'
            else:
                label = 'dog'
        else:
            label = 'neither'  # Uncertain - close to 0.5
        
        print(f"Prediction: {label}, Probability: {proba:.4f}, Confidence Margin: {confidence_margin:.4f}")

        # Save uploaded file for display
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.stream.seek(0)
        with open(save_path, 'wb') as f:
            f.write(file.stream.read())
        
        print(f"Image saved to: {save_path}")
        
        # Return result with prediction
        return render_template('index.html', result=label, prob=float(proba), img_path=save_path)
    
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return render_template('index.html', result=f'Error: {str(e)}'), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
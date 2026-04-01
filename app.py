from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import sys

app = Flask(__name__)
model = None

def load_model():
    global model
    try:
        from tensorflow.keras.models import load_model as keras_load
        if not os.path.exists("digit_model.h5"):
            print("ERROR: digit_model.h5 not found!")
            print("Please run: python train_model.py")
            sys.exit(1)
        model = keras_load("digit_model.h5")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_array = np.array(data['image'], dtype=np.float32)
        
        if image_array.shape != (784,):
            image_array = image_array.reshape(784)
        
        # Normalize if not already
        if image_array.max() > 1.0:
            image_array = image_array / 255.0

        image_array = image_array.reshape(1, 784)
        
        prediction = model.predict(image_array, verbose=0)
        probabilities = prediction[0].tolist()
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        # Top 3 predictions
        top3_indices = np.argsort(prediction[0])[::-1][:3].tolist()
        top3 = [
            {'digit': int(i), 'confidence': round(float(prediction[0][i]) * 100, 2)}
            for i in top3_indices
        ]

        return jsonify({
            'digit': digit,
            'confidence': round(confidence, 2),
            'probabilities': [round(p * 100, 2) for p in probabilities],
            'top3': top3
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    load_model()
    print("\n🚀 Server running at http://127.0.0.1:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)

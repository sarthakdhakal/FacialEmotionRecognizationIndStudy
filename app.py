from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from gtts import gTTS
from datetime import datetime
import os
import glob

app = Flask(__name__)

# Load model
try:
    model = load_model("emotion_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
class_names = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

def preprocess_frame(data_url):
    _, encoded = data_url.split(",", 1)
    decoded = base64.b64decode(encoded)
    img = Image.open(BytesIO(decoded)).convert("L") # Convert to grayscale
    img_np = np.array(img)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
        
    # Use the first detected face (or the largest one)
    (x, y, w, h) = faces[0]
    face_roi = img_np[y:y+h, x:x+w]
    
    # Resize to 48x48
    img_resized = cv2.resize(face_roi, (48, 48))
    
    # Normalize and expand dimensions
    img_processed = np.expand_dims(img_resized, axis=-1) # (48, 48, 1)
    img_processed = np.expand_dims(img_processed, axis=0) # (1, 48, 48, 1)
    img_processed = img_processed / 255.0
    
    return img_processed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'prediction': 'Model not loaded', 'audio': ''})

    data = request.get_json()
    image_data = data['image']

    try:
        img_array = preprocess_frame(image_data)
        
        if img_array is None:
             return jsonify({'prediction': 'No Face Detected', 'audio': ''})
             
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names.get(predicted_index, "Unknown")
        
        # Convert processed face back to base64 for display
        # img_array is (1, 48, 48, 1), normalized 0-1
        face_img = (img_array[0, :, :, 0] * 255).astype(np.uint8)
        pil_img = Image.fromarray(face_img)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        face_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        face_data_url = f"data:image/jpeg;base64,{face_b64}"

    except Exception as e:
        print("Prediction failed:", e)
        return jsonify({'prediction': 'Error', 'audio': ''})

    for file in glob.glob("static/prediction_*.mp3"):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    audio_filename = f"prediction_{timestamp}.mp3"
    audio_path = os.path.join("static", audio_filename)

    tts = gTTS(text=predicted_class)
    tts.save(audio_path)

    return jsonify({
        'prediction': predicted_class,
        'audio': f"/static/{audio_filename}",
        'face': face_data_url
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
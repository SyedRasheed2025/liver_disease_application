from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import os
import io
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)


model_fibrosis = load_model('liver_fibrosis_xception_model.h5')
class_labels = ['f0', 'f1f2f3', 'f4']
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model_detection = YOLO('liver_disease_model.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_liver_fibrosis(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model_fibrosis.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]
    
    if predicted_label == 'f0':
        return "No liver fibrosis detected."
    elif predicted_label == 'f1f2f3':
        return "Liver fibrosis detected at early stage."
    elif predicted_label == 'f4':
        return "Liver fibrosis detected in stage f4."
    else:
        return "Error in prediction."


def preprocess_image(image):
    image = np.array(image)
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/lcls', methods=['GET', 'POST'])
def lcls():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            prediction_result = predict_liver_fibrosis(filepath)
            
            return render_template('lcls.html', filename=filename, result=prediction_result)
    
    return render_template('lcls.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/ldec', methods=['GET', 'POST'])
def ldec():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)
            
            temp_image_path = "temp_uploaded_image.jpg"
            cv2.imwrite(temp_image_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            
            results = model_detection.predict(temp_image_path)
            
            image = cv2.imread(temp_image_path)
            for result in results:
                if hasattr(result, 'boxes'):
                    boxes = result.boxes
                    for box in boxes:
                        xyxy_np = box.xyxy.cpu().numpy().flatten()
                        x1, y1, x2, y2 = xyxy_np
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = result.names[class_id]
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"{class_name} ({confidence:.2f})"
                        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_image_path = "output_image.jpg"
            cv2.imwrite(output_image_path, image)
            
            return send_file(output_image_path, mimetype='image/jpeg')

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('ldec.html')

if __name__ == '__main__':
    app.run(debug=True)

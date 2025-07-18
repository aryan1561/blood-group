from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
import base64
import smtplib
from io import BytesIO
from PIL import Image
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('./model/model.h5')

# Email sender credentials (sender stays fixed)
EMAIL_SENDER = "pranathibathineni123@gmail.com"
EMAIL_PASSWORD = "qbbq iqjm jsaq xvks"

CLASS_NAMES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
UPLOAD_DIR = 'uploads'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

def preprocess_image(file_path):
    img = load_img(file_path, target_size=(64, 64))
    img_array = img_to_array(img)
    return np.expand_dims(img_array, axis=0)

def send_email(to_address, predicted_label, confidence, image_path=None):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = to_address
    msg['Subject'] = "Blood Group Detection Result"

    html_body = f"""
    <h2>Blood Group Detection Report</h2>
    <p><strong>Predicted Blood Group:</strong> {predicted_label}</p>
    <p><strong>Confidence:</strong> {confidence:.2f}</p>
    """

    msg.attach(MIMEText(html_body, 'html'))

    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read(), name=os.path.basename(image_path))
            msg.attach(img)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"✅ Email sent to {to_address}")
    except Exception as e:
        print(f"❌ Email send failed: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    email = request.form.get('email')

    if not file or not email:
        return jsonify({'error': 'Missing file or email'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    try:
        img = preprocess_image(file_path)
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_label = CLASS_NAMES[predicted_class]

        send_email(email, predicted_label, confidence, file_path)

        return jsonify({
            'predicted_label': predicted_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    data = request.get_json()
    if not data or 'image' not in data or 'email' not in data:
        return jsonify({'error': 'Missing image or email'}), 400

    try:
        base64_str = data['image'].split(',')[1]
        img_data = base64.b64decode(base64_str)

        file_path = os.path.join(UPLOAD_DIR, 'camera_image.png')
        with open(file_path, 'wb') as f:
            f.write(img_data)

        img = preprocess_image(file_path)
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_label = CLASS_NAMES[predicted_class]

        send_email(data['email'], predicted_label, confidence, file_path)

        return jsonify({
            'predicted_label': predicted_label,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)


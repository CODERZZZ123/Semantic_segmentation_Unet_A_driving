from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

# Load the trained U-Net model
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

# Function to preprocess images
def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (96*2, 128*2), method=tf.image.ResizeMethod.BILINEAR)
    img = img[tf.newaxis, ...]  # Add batch dimension
    return img

# Function to convert image to base64 string
def image_to_base64(img):
    img_pil = tf.keras.preprocessing.image.array_to_img(img[0])
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def adjust_brightness(image, factor=1.5):
    """
    Adjust the brightness of an image by scaling its pixel values.

    Parameters:
    - image: NumPy array representing the image.
    - factor: Brightness adjustment factor. Use values greater than 1 to increase brightness.

    Returns:
    - Adjusted image.
    """
    adjusted_image = image * factor
    adjusted_image = np.clip(adjusted_image, 0, 1)  # Clip values to the valid range [0, 1]
    return adjusted_image

# Route to handle image segmentation requests
@app.route('/segment', methods=['POST'])
def segment_image():
    # Get the uploaded image file from the request
    uploaded_file = request.files['file']

    # Save the uploaded file temporarily
    temp_path = 'temp_image.png'
    uploaded_file.save(temp_path)

    # Preprocess the uploaded image
    input_image = preprocess_image(temp_path)

    # Perform segmentation using the trained model
    pred_mask = model.predict(input_image)

    # Create a mask from model predictions
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    # Remove the temporary uploaded file
    os.remove(temp_path)
    
    # print(pred_mask_array)
    # Convert the original image to a base64 string
    original_image_base64 = image_to_base64(input_image)
    # resultPath_float = tf.cast(pred_mask,tf.float32)
    # pred_mask = adjust_brightness(pred_mask, factor=1.5)
    pred_image_base64 = image_to_base64(pred_mask)

    

    # Return the segmented mask and original image as JSON
    return jsonify({'segmented_mask_base64': pred_image_base64, 'original_image_base64': original_image_base64})

if __name__ == '__main__':
    app.run(debug=True)

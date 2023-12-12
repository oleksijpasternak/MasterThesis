from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file
    file.save('uploaded_photo.jpg')

    # Load the image
    image_path = 'uploaded_photo.jpg'
    original_image = cv2.imread(image_path)

    def get_model_result(model_path, img):
        model = load_model(model_path)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        # predictions = model.predict(img_array)

        print(predictions)
        return str(predictions[0][0])

    # Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the image at {image_path}")
    else:
    # Resize the image to 224 x 224 pixels
        target_size = (224, 224)
        img = cv2.resize(original_image, target_size)

        g_vgg_19_result = get_model_result("models/g_vgg_19_model.h5",img)
        armd_vgg_16_result = get_model_result("models/armd_vgg_16_model.h5",img)
        h_vgg_16_result = get_model_result("models/h_vgg_16_model.h5",img)
        c_vgg_16_result = get_model_result("models/c_vgg_16_model.h5",img)
        m_vgg_19_result = get_model_result("models/m_vgg_19_model.h5",img)
        d_vgg_16_result = get_model_result("models/d_vgg_16_model.h5",img)
        

        # Save the resized image
        resized_image_path = 'img_224.jpg'
        cv2.imwrite(resized_image_path, img)

        return "Glaucoma:"+g_vgg_19_result+" ARMD:" + armd_vgg_16_result+" Hypertension:" + h_vgg_16_result+" Cataract:" + c_vgg_16_result+" Myopia:" + m_vgg_19_result+" Diabetic:" + d_vgg_16_result

    
    return "Photo uploaded successfully!"

if __name__ == '__main__':
    app.run(debug=True)

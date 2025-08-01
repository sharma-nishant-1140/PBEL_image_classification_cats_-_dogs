from flask import Flask, render_template, url_for, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model("dog_cat_classifier.h5")

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    file = request.files["input_image"]
    if file.filename == "":
        return jsonify({"error" : "No file selected !"})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array)[0][0])
    label = "Dog" if prediction > 0.5 else "Cat"
    result = {"Predicted": label, "Confidence": prediction}

    return jsonify({"result": result, "image_name": file.filename})
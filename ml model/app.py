from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import requests
from PIL import Image
import numpy as np

app = Flask(__name__, static_url_path='/static', static_folder='C://Users//PMLS//Documents//project data//ml model//static' ,template_folder='C://Users//PMLS//Documents//project data//ml model//templates')

# Replace with your model loading logic (assuming some_ml_library provides a load_model function)
model = tf.keras.models.load_model("C://Users//PMLS//Documents//project data//ml model//temp_cnn.h5")
model.compile(optimizer='adam',loss='mse')

@app.route("/")
def splash():
    return render_template("splash.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get uploaded images and air temperature
        images = request.files.getlist("images")
        air_temp = int(request.form["air_temp"])
        image = Image.open(images)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        air_temp_input = np.array([[air_temp]])        # Make prediction
        prediction = model.predict([image_array, air_temp_input])

        return redirect(url_for("prediction", predictions=prediction))
    return render_template("upload.html")

@app.route("/prediction/<predictions>")
def prediction(predictions):
    # Convert comma-separated predictions to a list
    prediction_list = predictions.split(",")
    return render_template("prediction.html", predictions=prediction_list)

if __name__ == "__main__":
    app.run(debug=False, port='0.0.0.0')

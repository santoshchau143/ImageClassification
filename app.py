from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model("my_model.keras")

def prepare_image(image):
    image = image.resize((128,128))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET","POST"])
def home():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(file)
            img = prepare_image(image)
            pred = model.predict(img)[0][0]

            prediction = "Dog 🐶" if pred > 0.5 else "Cat 🐱"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
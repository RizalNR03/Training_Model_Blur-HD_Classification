from flask import Flask,jsonify, request
from werkzeug.utils import secure_filename
from tensorflow import keras
from keras.models import load_model 
import os
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/uploads/"

def allowed_file(filename):
    return '.' in filename and \
    filename.split('.', 1)[1] in app.config["ALLOWED_EXTENSIONS"]


model = load_model("model.h5", compile=False)
with open("labels.txt", "r") as file:
    labels = file.read().splitlines()


@app.route('/')
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200


@app.route('/prediction', methods=["GET", "POST"])
def predicition():
    if request.method == "POST":
        image = request.files['image']
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_img_array

            predicition = model.predict(data)
            index = np.argmax(predicition)
            class_names = labels[index]
            class_names = class_names[2:]
            confidence_score = predicition[0][index]


            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success Predicting the Image",
                },
                "data": {
                    "image_types_prediction": class_names,
                    "confidence": float(confidence_score)
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad Request",
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method Not Allowed",
            },
            "data": None
        }), 405



if __name__ == '__main__':
    app.run()

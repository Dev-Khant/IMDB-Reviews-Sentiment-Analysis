import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

new_model = load_model('cnn_model.h5')
app = Flask(__name__)

@app.route('/')
def home():

    return render_template('start.html')    

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    im = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(img_rgb, (224,224))
    im = np.expand_dims(im, axis=0)
    prediction = new_model.predict([im])
    if np.max(prediction) >= 0.5:
        pred = 'Car'
    else:
        pred = 'Bike'

    return render_template('second.html', prediction_text='It is a {a}'.format(a=pred))


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask,request,app,jsonify,url_for,render_template
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from PIL import Image 
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow import keras

model_now = tf.keras.models.load_model('Model.h5',compile = False)

model_now.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


print("Hello world")
app = Flask(__name__)

# model = load_model('Model.h5')

model_now.make_predict_function()

def predict_label(img_path):
        
        i = tf.keras.preprocessing.image.load_img(img_path, target_size=(128,128))
        i = tf.keras.preprocessing.image.img_to_array(i)/255.0
        i = i.reshape(1, 128,128,3)
        p = model_now.predict(i)
        return p
        

@app.route('/')
def home():
    return render_template("login.html")


@app.route('/LOGIN', methods = ['POST'])
def login():
    name1 = request.form['username']
    email = request.form['email']
    return render_template("index.html", name2 = name1)



@app.route('/predict', methods = ['POST'])
def predict():
    img = request.files['my_image']

    img_path = "static/" + img.filename
    img.save("img_path")

    text1 = """Casting having smooth finishing,
                ready for production purpose,
                tight tolerances, Affordable 
                Tooling which allows for costs 
                to remain low, vast size range
                and variety of materials that 
                can be used."""


    text2 = """Due to an undesired irregularity 
              in a metal casting process like
              Gas Porosity: Blowholes, open 
              holes,pinholes. Shrinkage defects:
              shrinkage cavity. Pouring metal 
              defects: Cold shut, misrun, slag 
              inclusion."""
   
    input_prediction = predict_label(img_path)

    if input_prediction >= 0.65:

        return render_template("thanku.html", prediction_text = "Casting is OK", img_path = img_path, text = text1)
    
    else:

         return render_template("thanku.html", prediction_text = "Casting is Defective", img_path = img_path, text = text2)
    
if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)    
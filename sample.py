import cv2
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow import keras
from PIL import Image 


model_now = tf.keras.models.load_model('Model.h5',compile = False)

model_now.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)


input_image_resized = cv2.resize(input_image, (128,128))

input_image_scaled = input_image_resized/255

input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])

input_prediction = model_now.predict(input_image_reshaped)

print(input_prediction)


input_pred_label = np.argmax(input_prediction)

print(input_pred_label)


if input_prediction >= 0.65:

  print('Cating is Ok')

else:

  print('Casting is defective')
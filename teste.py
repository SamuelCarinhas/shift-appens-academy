from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import sys

image_path = sys.argv[1]

img = cv2.imread(image_path)
image = tf.image.resize(img, (256, 256))


model = load_model('models/classifier.keras')

if model.predict(np.expand_dims(image/255, 0), 0) > 0.5:
    print(f'Predicted class is First')
else:
    print(f'Predicted class is Second')


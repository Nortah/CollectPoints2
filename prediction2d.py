# TensorFlow and tf.keras
import csv

import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
import DataCollect as dc

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# get labels
labels = []
with open("labels.csv") as f:
    reader = csv.reader(f)
    for row in reader: # each row is a list
        labels.append(row)
print(labels)
inputsData = dc.DataCollect('C:/Users/Nestor/Documents/Travail de Bachelor/Input2d/')
inputs = inputsData.get_data()
model = tf.keras.models.load_model('./my_model')
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
img = inputs[0]
img = (np.expand_dims(img, 0))
prob = probability_model.predict(img)
index = prob.argmax(axis=-1)
print(labels[index[0]])

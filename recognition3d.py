# TensorFlow and tf.keras
import tensorflow as tf
import DataCollect as dc

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
img_height = 480
img_width = 640
# Import data
train_data = dc.DataCollect('C:/Users/Nestor/Documents/Travail de Bachelor/3dTraining/', 'D')
train_images = train_data.get_data()
test_data = dc.DataCollect('C:/Users/Nestor/Documents/Travail de Bachelor/3dTest/', 'D')
test_images = test_data.get_data()

# class names dictionary
class_names = np.array(train_data.label_values)
class_names = np.unique(class_names)

# save class names
np.savetxt('./labels3d.csv', class_names, fmt='%s')

# Get labels
train_labels = np.array([], dtype='int')
for item in train_data.label_values:
    train_labels = np.append(train_labels, np.where(class_names == item))

test_labels = np.array([], dtype='int')
for item in test_data.label_values:
    test_labels = np.append(test_labels, np.where(class_names == item))


print(train_images.shape)
print(train_labels)
print(test_images.shape)
print(test_labels)

# fashion_mnist = tf.keras.datasets.fashion_mnist
#
# train_labels
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class names dictionary

# normalize images between 0 and 1
# train_images = train_images / 3000.0
print(train_images.shape)
print(train_labels)
# test_images = test_images / 3000.0
print(test_images.shape)
print(test_labels)

# create model
model = tf.keras.Sequential([
    # data_augmentation,
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(class_names.__len__())
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

model.save('./model3d')
print('\nTest accuracy:', test_acc)

# Create a probability array
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
print(probability_model)
# Does a prediction for each images in the test dataset
predictions = probability_model.predict(test_images)
# results of the first image prediction
print(predictions[0])


# Graphical representation of predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(class_names.__len__()))
    plt.yticks([])
    thisplot = plt.bar(range(class_names.__len__()), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 4
num_cols = 4
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
# ----------------------- Prediction for 1 item ----------------------/
# Grab an image from the test dataset.
# img = test_images[2]
#
# # Add the image to a batch where it's the only member.
# img = (np.expand_dims(img, 0))
#
# predictions_single = probability_model.predict(img)
#
# print(predictions_single)
#
# plot_value_array(1, predictions_single[0], test_labels)
# _ = plt.xticks(range(class_names.__len__()), class_names, rotation=45)
# plt.show()
# ----------------------- Prediction for 1 item ----------------------/

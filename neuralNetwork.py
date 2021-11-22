import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyrealsense2 as rs
import os
# set max size of numpy
# np.set_printoptions(threshold=sys.maxsize)

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.asarray(np.random.randint(0, 3000, 307200), dtype=np.float_)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
                derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors


class ReadBag:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # size of the array is 480*640 as it's the resolution
        self.training_array = np.empty((0, 307200), int)

    def getTrainingData(self):
        files = os.listdir(self.folder_path)
        for f in files:
            if f.endswith(".bag"):
                print(f)
                try:
                    # Create pipeline
                    pipeline = rs.pipeline()

                    # Create a config object
                    config = rs.config()

                    # Tell config that we will use a recorded device from file to be used by the pipeline through
                    # playback.
                    rs.config.enable_device_from_file(config, self.folder_path + f)

                    # Configure the pipeline to stream the depth stream
                    # Change this parameters according to the recorded bag file resolution
                    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

                    # Start streaming from file
                    pipeline.start(config)

                    # Get frameset of depth
                    frames = pipeline.wait_for_frames()

                    # Get depth frame
                    depth_frame = frames.get_depth_frame()

                    array = np.asanyarray(depth_frame.get_data())
                    array = array.flatten()
                    self.training_array = np.vstack((self.training_array, array))

                finally:
                    pass
        return self.training_array


# Get training data
right_training_data = ReadBag("C:/Users/Nestor/Documents/Travail de Bachelor/Training_dataset/Right_Data/")
false_training_data = ReadBag("C:/Users/Nestor/Documents/Travail de Bachelor/Training_dataset/Wrong_Data/")
# Get Input data
input_vector_false = ReadBag("C:/Users/Nestor/Documents/Travail de Bachelor/Inputs/Wrong_input/")
input_vector_false = input_vector_false.getTrainingData().flatten()
input_vector_right = ReadBag("C:/Users/Nestor/Documents/Travail de Bachelor/Inputs/Right_input/")
input_vector_right = input_vector_right.getTrainingData().flatten()
print(input_vector_right.shape)
targets = np.array([], dtype=int)
# Set input vectors with all false values
input_vectors = false_training_data.getTrainingData()
input_vectors = np.array(input_vectors)
print(input_vectors.shape)
false_data_rows, colums = input_vectors.shape
for x in range(false_data_rows):
    targets = np.append(targets, int(0))

# Set input vectors with all false values
input_vectors_temp = right_training_data.getTrainingData()
right_data_rows, colums = input_vectors_temp.shape
for x in range(right_data_rows):
    targets = np.append(targets, int(1))


print("Start concat ")
# Merge the two training vectors
input_vectors = np.concatenate((input_vectors, input_vectors_temp), axis=0)
print("End concat ")
print(input_vectors.shape)
print(targets)

targets = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0]

learning_rate = 0.1

print("Create object")
neural_network = NeuralNetwork(learning_rate)
print("prediction start")
print("false array result: ", neural_network.predict(input_vector_false))
print("right array result: ", neural_network.predict(input_vector_right))

print("Training start")
training_error = neural_network.train(input_vectors, targets, 1000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")

print("false array result: ", neural_network.predict(input_vector_false))
print("right array result: ", neural_network.predict(input_vector_right))



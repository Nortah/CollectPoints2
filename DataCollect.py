import os

import cv2
import numpy as np
import pyrealsense2 as rs
import pretreatment as pr

class DataCollect:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # size of the array is 480*640 as it's the resolution
        self.array = np.empty((0, 480, 640), int)
        self.label_values = []

    def get_data(self):
        files = os.listdir(self.folder_path)
        for f in files:
            person = f[0: f.find(';')]
            self.label_values.append(person)
            if f.endswith(".jpg"):
                try:
                    treat_image = pr.Pretreatment(self.folder_path + f)
                    image = treat_image.get_roi()

                    # image = cv2.imread(self.folder_path + f)
                    print(image.shape)
                    # Convert image to grayscale. The second argument in the following step is cv2.COLOR_BGR2GRAY, which converts colour
                    # image to grayscale.
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.array = np.concatenate((self.array, [gray]))
                finally:
                    pass
            elif f.endswith(".bag"):
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
                    # Get color frames
                    color_frame = frames.get_color_frame()
                    # Get depth frame
                    depth_frame = frames.get_depth_frame()
                    array = np.asanyarray(depth_frame.get_data())
                    print(array.shape)
                    self.array = np.concatenate((self.array, [array]))



                finally:
                    pass

        return self.array


# train_images = DataCollect('C:/Users/Nestor/Documents/Travail de Bachelor/2dTest/')
# # Get labels
# train_labels = np.array([])
#
#
# print(train_labels)
# train_images_ = train_images.get_data()
#
# for item in train_images.label_values:
#     print(train_images.label_values)
#     result = train_images.label_values.index(item)
#     print(result)
#     train_labels = np.append(train_labels, result)
#     print(train_labels)
# print(train_labels)

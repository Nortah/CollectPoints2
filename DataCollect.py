import os

import cv2
import numpy as np


class DataCollect:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # size of the array is 480*640 as it's the resolution
        self.array = np.empty((0, 480, 640), int)

    def get_data(self):
        files = os.listdir(self.folder_path)
        for f in files:
            if f.endswith(".jpg") or f.endswith(".bag"):
                try:
                    image = cv2.imread(self.folder_path + f)

                    # Convert image to grayscale. The second argument in the following step is cv2.COLOR_BGR2GRAY, which converts colour
                    # image to grayscale.
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    self.array = np.concatenate((self.array, [gray]))


                finally:
                    pass
        return self.array


train_images = DataCollect('C:/Users/Nestor/Documents/Travail de Bachelor/2dTraining/')
train_images = train_images.get_data()
print(train_images.shape)

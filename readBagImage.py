import subprocess
import yaml
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np


FILENAME = 'Indoor'
ROOT_DIR = '/home/Dataset'
BAGFILE = ROOT_DIR + '/' + FILENAME + '.bag'

if __name__ == '__main__':
    bag = rosbag.Bag(BAGFILE)
    for i in range(2):
        if (i == 0):
            TOPIC = '/camera/depth/image_rect_raw'
            DESCRIPTION = 'depth_'
        else:
            TOPIC = '/camera/color/image_raw'
            DESCRIPTION = 'color_'
        image_topic = bag.read_messages(TOPIC)
        for k, b in enumerate(image_topic):
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(b.message, b.message.encoding)
            cv_image.astype(np.uint8)
            if (DESCRIPTION == 'depth_'):
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(ROOT_DIR + '/depth/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
            else:
                cv2.imwrite(ROOT_DIR + '/color/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
            print('saved: ' + DESCRIPTION + str(b.timestamp) + '.png')


    bag.close()

    print('PROCESS COMPLETE')
# Reading an image file into Python as a PIL image is straightforward as shown in the code snippet below. This reads an image, in the given root directory, named ‘color_002482024802.png’
#
# from skimage import io
# import os
# import pandas as pd
# import numpy as np
# from PIL import Image
#
# root_dir = '/home/Dataset/color'
# colorimstring = 'color_002482024802.png'
# colorimage = io.imread(os.path.join(root_dir, colorimstring))
# colorimage_rgb = colorimage[:, :, [2, 1, 0]]
# colorim = Image.fromarray(colorimage_rgb)
# colorim.show()
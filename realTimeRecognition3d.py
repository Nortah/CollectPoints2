import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image as im
import pretreatment as pr
import csv
import tensorflow as tf

# ============================= Get model for prediction =======================================/
# get labels
labels = []
with open("labels3d.csv") as f:
    reader = csv.reader(f)
    for row in reader:  # each row is a list
        labels.append(row)
model = tf.keras.models.load_model('./model3d')
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
# ============================= Get model for prediction =======================================/
# ============================= Format Of Text to insert =======================================/
org = (250, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

color = (0, 255, 0)  # (B, G, R)
thickness = 3
lineType = cv2.LINE_AA
bottomLeftOrigin = False

# ============================= Format Of Text to insert =======================================/
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

photoTaken = False
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # transform into image
        treated_image = pr.Pretreatment(color_image)
        facesCount = treated_image.get_faces_count()
        # if there's a face try to detect if it correspond to someone
        if facesCount > 0:
            image = (np.expand_dims(depth_image, 0))
            prob = probability_model.predict(image)
            index = prob.argmax(axis=-1)
            person = labels[index[0]]
            print(person)
            color_image = cv2.putText(color_image, str(person), org, font, fontScale, color,
                                      thickness, lineType, bottomLeftOrigin)

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)
        # Enter key
        if key == 13:
            print("in if)")
            cv2.destroyAllWindows()


finally:

    # Stop streaming
    pipeline.stop()

import numpy as np
import cv2

# Parameters
blur = 21
canny_low = 15
canny_high = 150
min_area = 0.0005
max_area = 0.95
dilate_iter = 10
erode_iter = 10
mask_color = (0.0,0.0,0.0)

# initialize video from the webcam
video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()

    if ret == True:
        # Convert image to grayscale
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny Edge Dection
        edges = cv2.Canny(image_gray, canny_low, canny_high)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        # get the contours and their areas
        contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]]

        # Get the area of the image as a comparison
        image_area = frame.shape[0] * frame.shape[1]

        # calculate max and min areas in terms of pixels
        max_area = max_area * image_area
        min_area = min_area * image_area

        # Set up mask with a matrix of 0's
        mask = np.zeros(edges.shape, dtype = np.uint8)

        # Go through and find relevant contours and apply to mask
        for contour in contour_info:
            # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
            if contour[1] > min_area and contour[1] < max_area:
                # Add contour to mask
                mask = cv2.fillConvexPoly(mask, contour[0], (255))
            # use dilate, erode, and blur to smooth out the mask
            mask = cv2.dilate(mask, None, iterations=dilate_iter)
            mask = cv2.erode(mask, None, iterations=erode_iter)
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)
            # Ensures data types match up
            mask_stack = mask_stack.astype('float32') / 255.0
            frame = frame.astype('float32') / 255.0
            # Use the q button to quit the operation
            if cv2.waitKey(60) & 0xff == ord('q'):
                break
        else:
            break
        cv2.destroyAllWindows()
        video.release()
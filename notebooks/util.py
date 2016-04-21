import cv2
import math
import pylab
import datetime
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

_begin = 0
total_time = datetime.timedelta()

# Time measurement functions

def begin():
    global _begin
    _begin = datetime.datetime.now()
    
def end():
    global _begin, total_time
    end = datetime.datetime.now() - _begin
    total_time += end
    print('Time: %s' % end)

def total():
    global total_time
    print('Total time: %s' % total_time)

# Plot image functions

def plot_image(image):
    if len(image.shape) == 2:
        image = cv2.merge((image,image,image))
    f, axarr = plt.subplots(1, 1, figsize=(10,15))
    axarr.axis('off')
    axarr.imshow(image)
    axarr.plot()
    
def plot_images(images):
    f, axarr = plt.subplots(1, len(images), figsize=(15,15))
    for i in range(len(images)):
        if len(images[i].shape) == 2:
            image = cv2.merge((images[i],images[i],images[i]))
        else:
            image = images[i]
        axarr[i].axis('off')
        axarr[i].imshow(image)
        axarr[i].plot()
        
# Load image function

def load_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pattern parameters
rows = 6
columns = 11
square_width = 13

# Pattern detection functions

def pattern_detection(image):
    # Convert image to 1 channel
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (columns, rows), flags=cv2.CALIB_CB_FAST_CHECK)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find corners with subpixel accuracy
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    return corners, ret

def draw_pattern(image, corners, ret):
    # Draw corners into image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.drawChessboardCorners(image, (columns, rows), corners, ret)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

import cv2
import math
import pylab
import struct
import datetime
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        image = cv2.merge((image, image, image))
    f, axarr = plt.subplots(1, 1, figsize=(10, 15))
    axarr.axis('off')
    axarr.imshow(image)
    axarr.plot()


def plot_images(images):
    f, axarr = plt.subplots(1, len(images), figsize=(15, 15))
    for i in range(len(images)):
        if len(images[i].shape) == 2:
            image = cv2.merge((images[i], images[i], images[i]))
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


# Read PLY functions

class Mesh(object):
    def __init__(self):
        self.vertexes = None
        self.colors = None
        self.normal = None
        self.vertex_count = 0


def _load_binary(mesh, stream, dtype, count):
    data = np.fromfile(stream, dtype=dtype, count=count)

    fields = dtype.fields
    mesh.vertex_count = count

    if 'v' in fields:
        mesh.vertexes = data['v']
    else:
        mesh.vertexes = np.zeros((count, 3))

    if 'n' in fields:
        mesh.normal = data['n']
    else:
        mesh.normal = np.zeros((count, 3))

    if 'c' in fields:
        mesh.colors = data['c']
    else:
        mesh.colors = 255 * np.ones((count, 3))


def load_ply(filename):
    m = Mesh()
    with open(filename, "rb") as f:
        dtype = []
        count = 0
        format = None
        line = None
        header = ''

        while line != 'end_header\n' and line != '':
            line = f.readline()
            header += line
        # Discart faces
        header = header.split('element face ')[0].split('\n')

        if header[0] == 'ply':

            for line in header:
                if 'format ' in line:
                    format = line.split(' ')[1]
                    break

            if format is not None:
                if format == 'ascii':
                    fm = ''
                elif format == 'binary_big_endian':
                    fm = '>'
                elif format == 'binary_little_endian':
                    fm = '<'

            df = {'float': fm + 'f', 'uchar': fm + 'B'}
            dt = {'x': 'v', 'nx': 'n', 'red': 'c', 'alpha': 'a'}
            ds = {'x': 3, 'nx': 3, 'red': 3, 'alpha': 1}

            for line in header:
                if 'element vertex ' in line:
                    count = int(line.split('element vertex ')[1])
                elif 'property ' in line:
                    props = line.split(' ')
                    if props[2] in dt.keys():
                        dtype = dtype + [(dt[props[2]], df[props[1]], (ds[props[2]],))]

            dtype = np.dtype(dtype)

            if format is not None:
                if format == 'binary_big_endian' or format == 'binary_little_endian':
                    _load_binary(m, f, dtype, count)
            return m
        else:
            return None

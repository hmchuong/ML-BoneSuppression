import numpy as np
import tensorflow as tf
import os
import cv2
from PIL import Image
from scipy.misc import imresize

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def sobel_conv(images, dim=5):
    sobel_x = tf.constant([
            [1, 0, -2, 0, 1],
            [4, 0, -8, 0, 4],
            [6, 0, -12, 0, 6],
            [4, 0, -8, 0, 4],
            [1, 0, -2, 0, 1]
        ], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [dim, dim, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    filtered_x = tf.nn.conv2d(images, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(images, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered = tf.sqrt(tf.pow(filtered_x, 2) + tf.pow(filtered_y, 2))
    return filtered

def extract_dicom(files, invert):
    images = []
    # loop through all the DICOM files
    for i, filenameDCM in enumerate(files):
        print("Extract: " + filenameDCM + " ", i)
        # read the jpg file
        ds = cv2.imread(filenameDCM)
        ds = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
        if invert:
            ds = cv2.bitwise_not(ds)
        images += [ds]
    return images

def extract_data(paths, num = -1, invert=False, extension="jpg"):
    lstFilesDCM = []  # create an empty list

    for path in paths:
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if "." + extension in filename.lower():
                    lstFilesDCM.append(os.path.join(dirName,filename))

    num = min(len(lstFilesDCM), num)
    if num == -1:
        num = len(lstFilesDCM)

    images = extract_dicom(sorted(lstFilesDCM)[:num], invert=invert)
    return images

def crop_to_square(image, upsampling):
    if image.shape[0] == image.shape[1]:
        return image
    if upsampling:
        img = Image.fromarray(image)
        target_side = max(img.size)
        horizontal_padding = (target_side - img.size[0]) / 2
        vertical_padding = (target_side - img.size[1]) / 2
        start = [-horizontal_padding, -vertical_padding]
        width = img.size[0] + horizontal_padding
        height = img.size[1] + vertical_padding
    else:
        target_side = min(image.shape)
        horizontal_padding = int((image.shape[0] - target_side) / 2)
        vertical_padding = int((image.shape[1] - target_side) / 2)
        start = [horizontal_padding, vertical_padding]
        width = image.shape[0] - horizontal_padding
        height = image.shape[1] - vertical_padding
        return image[start[0]:width, start[1]:height]

    img = img.crop((start[0], start[1], width, height))
    return np.array(img)

def preprocess(images, upsampling=False):
    images = [(im + abs(im.min())) / (im.max() + abs(im.min()))  for im in images]
    return images

def resize(images, size):
    return [imresize(i, (size,size), "lanczos") for i in images]

def crop(images, upsampling=False):
    return [crop_to_square(im, upsampling=upsampling) for im in images]

def checkAndCreateDir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

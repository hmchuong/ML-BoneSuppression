import numpy as np
import tensorflow as tf
import os
import cv2
from scipy.misc import imresize
from PIL import Image, ImageOps
import random
import sys
from sklearn.utils import shuffle

def crop_to_square(image, upsampling):
    """
    Crop image to square
    """
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

def extract_n_preprocess_dicom(path, size):
    """
    Extract DICOM image from path with preprocessing to size
    """
    ds = cv2.imread(path)
    ds = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
    ds = crop_to_square(ds, upsampling=True)
    ds = imresize(ds, (size,size), "lanczos")
    return ds

def extract_image(path):
    """
    Extract DICOM image from path
    """
    ds = cv2.imread(path)
    ds = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
    return ds

def augment_image_pair(image1, image2, size, output_path1, output_path2):
    """
    Augment image pair
    """
    image1 = Image.fromarray(image1).convert('L')
    image2 = Image.fromarray(image2).convert('L')

    offset = random.randint(0, 100)
    rotate = random.randint(-30,30)
    min_val = random.randint(0, offset+1)

    # Flip
    if random.randint(1,3) % 2 == 0:
        image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
        image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
    # Add offset
    image1 = ImageOps.expand(image1, offset)
    image2 = ImageOps.expand(image2, offset)
    # Rotate
    image1 = image1.rotate(rotate)
    image2 = image2.rotate(rotate)
    # Crop
    image1 = image1.crop((min_val, min_val, min_val+size, min_val+size))
    image2 = image2.crop((min_val, min_val, min_val+size, min_val+size))
    # Save
    image1.save(output_path1)
    image2.save(output_path2)

def extract_images(paths):
    """
    Extract images from paths
    """
    images = []
    for path in paths:
        ds = cv2.imread(path)
        ds = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
        images.append(ds)
    return images

def check_and_create_dir(dir_path):
    """
    Check and create directory path
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def extract_image_path(folders, extension="png"):
    """
    Extract image paths with extension from folders
    """
    images = []
    for folder in folders:
        for dirName, subdirList, fileList in os.walk(folder):
            for filename in fileList:
                if "." + extension in filename.lower():
                    images.append(os.path.join(dirName,filename))
    return images

def extract_n_normalize_image(path):
    """
    Extract DICOM image from path
    """
    ds = cv2.imread(path)
    ds = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
    return ds.astype(float)/255

def get_batch(batch_size, size, x_filenames, y_filenames):
    X, y = shuffle(x_filenames, y_filenames)
    X = X[:batch_size]
    y = y[:batch_size]
    X_images = []
    y_images = []
    for i in range(len(X)):
        X_images.append(extract_n_normalize_image(X[i]))
        y_images.append(extract_n_normalize_image(y[i]))
    X_images = np.reshape(np.array(X_images), (batch_size, size, size, 1))
    y_images = np.reshape(np.array(y_images), (batch_size, size, size, 1))
    return (X_images, y_images)

def print_train_steps(current_step, total_steps):
    point = int(current_step / (total_steps * 0.05))
    sys.stdout.write("\r[" + "=" * point +  " " * (20 - point) + "] ---- Step {}/{} ----- ".format(current_step, total_steps) +  str(int(float(current_step) * 100 / total_steps)) + "%")
    sys.stdout.flush()

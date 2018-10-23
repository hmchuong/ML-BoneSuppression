from __future__ import division
from configparser import ConfigParser
import argparse
from utils import extract_image_path, extract_n_preprocess_dicom, check_and_create_dir, extract_image, augment_image_pair
import imreg_dft as ird
import os
import cv2
from scipy.misc import imsave
from multiprocessing.pool import Pool
from itertools import product
import numpy as np

def similarity(params):
    """
    Transform target image to make it similar to source image
    """
    # The template
    source_image, target_image, size, source_out_dir, target_out_dir, verbose = params
    im0 = extract_n_preprocess_dicom(source_image, size)
    # The image to be transformed
    im1 = extract_n_preprocess_dicom(target_image, size)

    filename = os.path.basename(os.path.normpath(source_image))
    if verbose: print("Comparing %r .........." % filename)
    # Transform im1 to im0
    result = ird.similarity(im0, im1, numiter=3)
    source_image_path = os.path.join(source_out_dir, filename)
    target_image_path = os.path.join(target_out_dir, filename)
    if verbose: print("Saving %r .........." % source_image_path)
    imsave(source_image_path, im0)
    imsave(target_image_path, result['timg'])
    if verbose: print("Saved %r .........." % source_image_path)

def get_image_path_from(source_dir, target_dir):
    """
    Get image paths from source and target directory
    """
    source_images = extract_image_path([source_dir])
    target_images = extract_image_path([target_dir])

    assert len(source_images) == len(target_images), "Number of images in %r is not the same as %r" % (source_dir, target_dir)
    return (source_images, target_images)

def check_n_create_output_dir(output_dir):
    """
    Check and create output directory
    """
    source_out_dir = os.path.join(output_dir, "source")
    target_out_dir = os.path.join(output_dir, "target")
    check_and_create_dir(source_out_dir)
    check_and_create_dir(target_out_dir)
    return (source_out_dir, target_out_dir)

def registration(verbose, num_threads, size, source_dir, target_dir, output_dir):
    """
    Registrating images and save to output_dir
    """
    if verbose: print("Get image paths ...")
    # Get images paths
    source_images, target_images = get_image_path_from(source_dir, target_dir)

    # Check and create output directory
    source_out_dir, target_out_dir = check_n_create_output_dir(output_dir)

    pool = None
    if num_threads >= 1:
        pool = Pool(num_threads)
    else:
        pool = Pool()
    job_args = [(source_images[i], target_images[i], size, source_out_dir, target_out_dir, verbose) for i in range(len(source_images))]
    pool.map(similarity, job_args)
    pool.close()
    pool.join()

def augmentation_pair(params):
    """
    Augmenting pair of images
    """
    source_image_path, target_image_path, size, source_out_path, target_out_path, verbose = params
    source_image = extract_image(source_image_path)
    target_image = extract_image(target_image_path)
    filename = os.path.basename(os.path.normpath(source_image_path))
    if verbose:
        print("Augmenting image %r ..." % filename)
    augment_image_pair(source_image, target_image, size, source_out_path, target_out_path)
    if verbose:
        print("Augmented image %r ..." % filename)

def augmentation(verbose, num_threads, source_dir, target_dir, augmentation_seed, size, output_dir):
    """
    Augment registered images and save to output_dir
    """
    # Get images paths
    if verbose: print("Get image paths ...")
    source_images, target_images = get_image_path_from(source_dir, target_dir)

    # Check and create output directory
    source_out_dir, target_out_dir = check_n_create_output_dir(output_dir)

    # Augmenting images
    pool = None
    if num_threads >= 1:
        pool = Pool(num_threads)
    else:
        pool = Pool()
    job_args = []
    for seed in range(augmentation_seed):
        for i in range(len(source_images)):
            image_name = '%r_%r.png' % (seed, i)
            source_out_path = os.path.join(source_out_dir, image_name)
            target_out_path = os.path.join(target_out_dir, image_name)
            job_args.append((source_images[i], target_images[i], size, source_out_path, target_out_path, verbose))
    pool.map(augmentation_pair, job_args)
    pool.close()
    pool.join()

def main(args):
    cp = ConfigParser()
    cp.read(args.config)

    verbose = cp["DATA"].getboolean("verbose")
    num_threads = cp["DATA"].getint("num_threads")
    image_size = cp["DATA"].getint("image_size")

    # Data registration
    source_dir = cp["REGISTRATION"].get("source_dir")
    target_dir = cp["REGISTRATION"].get("target_dir")
    output_dir = cp["REGISTRATION"].get("registered_output_dir")
    is_registration = cp["REGISTRATION"].getboolean("data_registration")
    if is_registration:
        if verbose: print("Starting registration data ...")
        registration(verbose, num_threads, image_size, source_dir, target_dir, output_dir)

    # Data augmentation
    source_dir = cp["AUGMENTATION"].get("source_dir")
    target_dir = cp["AUGMENTATION"].get("target_dir")
    output_dir = cp["AUGMENTATION"].get("augmented_output_dir")
    augmentation_seed = cp["AUGMENTATION"].getint("augmentation_seed")
    is_augmentation = cp["AUGMENTATION"].getboolean("data_augmentation")
    if is_augmentation:
        if verbose: print("Starting augmentation data ...")
        augmentation(verbose, num_threads, source_dir, target_dir, augmentation_seed, image_size, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hmchuong - BoneSuppression v2 - Preprocessing data')
    parser.add_argument('--config', default='config/data_preprocessing.cfg', type=str, help='config file')
    args = parser.parse_args()
    main(args)

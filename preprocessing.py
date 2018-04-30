from utils import extract_data, checkAndCreateDir, resize, crop, preprocess
import os
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave, imresize
import imreg_dft as ird
import numpy as np

class ImageProcessing:
    """ImageProcessing for registration, augmentation images"""
    def __init__(self, image_size):
        self.image_size = image_size

    def registration(self, jsrt_source_dir, need_invert, bse_jsrt_source_dir, output_dir):
        """
        Registrating images and save to output_dir
        """
        x_images = extract_data([jsrt_source_dir], invert=need_invert)
        y_images = extract_data([bse_jsrt_source_dir])

        # Resize images
        x_images = resize(x_images, self.image_size)

        # Check output directory
        checkAndCreateDir(output_dir)
        x_dir = os.path.join(output_dir, "x")
        y_dir = os.path.join(output_dir, "y")
        checkAndCreateDir(x_dir)
        checkAndCreateDir(y_dir)

        for i in range(len(x_images)):
            # the template
            im0 = x_images[i]
            # the image to be transformed
            im1 = imresize(y_images[i], im0.shape, 'lanczos')
            result = ird.similarity(im0, im1, numiter=3)
            x_image_path = os.path.join(x_dir,'bs_' + str(i) + '.jpg')
            y_image_path = os.path.join(y_dir,'bs_' + str(i) + '.jpg')
            imsave(x_image_path, x_images[i])
            imsave(y_image_path, result['timg'])

    def augmentation(self, input_dir, output_dir, seed):
        """
        Augmentation data
        """
        # Input
        x_path = os.path.join(input_dir, "x")
        y_path = os.path.join(input_dir, "y")
        if (not os.path.exists(x_path)) or (not os.path.exists(y_path)):
            raise Exception('Sub directory "x" and "y" do not exist in {}'.format(input_dir))

        x_images = extract_data([x_path])
        y_images = extract_data([y_path])

        # Output
        x_path = os.path.join(output_dir, "x")
        y_path = os.path.join(output_dir, "y")
        checkAndCreateDir(x_path)
        checkAndCreateDir(y_path)

        train = crop(x_images, upsampling='True')
        train = resize(train, self.image_size)
        train = preprocess(train)
        trainX = np.reshape(train, (len(train), self.image_size, self.image_size, 1))

        groundtruth = crop(y_images, upsampling='True')
        groundtruth = resize(groundtruth, self.image_size)
        groundtruth = preprocess(groundtruth)
        trainY = np.reshape(groundtruth, (len(groundtruth), self.image_size, self.image_size, 1))

        data_gen_args = dict(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=5.,
                    width_shift_range=0.08,
                    height_shift_range=0.08,
                    shear_range=0.06,
                    zoom_range=0.08,
                    channel_shift_range=0.2,
                    fill_mode='constant',
                    cval=0.,
                    horizontal_flip=True,
                    vertical_flip=False,
                    rescale=None)
        image_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen.fit(trainX, augment=True, seed=1)
        batch_size = len(trainX)
        for i in range(seed):
            print("Generate seed: {}/{}".format(i+1,seed))
            x = image_datagen.flow(trainX, shuffle=True, seed=i, save_format='jpeg', save_to_dir=x_path, batch_size=batch_size)
            y = image_datagen.flow(trainY, shuffle=True, seed=i, save_format='jpeg', save_to_dir=y_path, batch_size=batch_size)
            _ = x.next()
            _ = y.next()

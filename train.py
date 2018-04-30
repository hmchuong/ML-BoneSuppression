from configparser import ConfigParser
from preprocessing import ImageProcessing

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # Parse arguments
    is_registration = cp["DATA"].getboolean("image_registration")
    registered_images_dir = cp["DATA"].get("registered_images_dir")
    image_size = int(cp["DATA"].getint("image_size"))
    is_augment_data = cp["DATA"].getboolean("augment_data")

    image_processing = ImageProcessing(image_size)

    # Images registration
    if is_registration:
        jsrt_source_dir = cp["DATA"].get("jsrt_source_dir")
        need_invert = cp["DATA"].getboolean("need_invert")
        bse_jsrt_source_dir = cp["DATA"].get("bse_jsrt_source_dir")
        image_processing.registration(jsrt_source_dir, need_invert, bse_jsrt_source_dir, registered_images_dir)

    # Images augmentation
    if is_augment_data:
        augmentation_seed = cp["DATA"].getint("augmentation_seed")
        augmented_images_dir = cp["DATA"].get("augmented_images_dir")
        image_processing.augmentation(registered_images_dir, augmented_images_dir, augmentation_seed)

    # Training
    


if __name__ == '__main__':
    main()

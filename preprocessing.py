from utils import extract_data

class ImageProcessing:
    """ImageProcessing for registration, augmentation images"""
    def __init__(self, jsrt_source_dir, need_invert, bse_jsrt_source_dir, image_size):
        self.jsrt_source_dir = jsrt_source_dir
        self.need_invert = need_invert
        self.bse_jsrt_source_dir = bse_jsrt_source_dir
        self.image_size = image_size

    def registration(self, output_dir):
        """
        Registrating images and save to output_dir
        """
        x_images = extract_data([self.jsrt_source_dir])
        y_images = extract_data([self.bse_jsrt_source_dir])

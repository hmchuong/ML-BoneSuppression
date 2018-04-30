from configparser import ConfigParser
from model import AELikeModel

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # Parse arguments
    image_size = cp["DATA"].getint("image_size")
    trained_model = cp["TEST"].get("model")
    input_dir = cp["TEST"].get("input_dir")
    need_invert = cp["TEST"].getboolean("need_invert")
    output_dir = cp["TEST"].get("output_dir")
    alpha = cp["TRAIN"].getfloat("alpha")

    # Test
    model = AELikeModel(image_size, alpha, trained_model)
    model.test(input_dir, output_dir, need_invert)

if __name__ == '__main__':
    main()

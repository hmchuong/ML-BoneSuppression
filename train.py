from __future__ import division
from configparser import ConfigParser
import argparse
from model import AELikeModel

def main(args):
    # parser config
    cp = ConfigParser()
    cp.read(args.config)

    # Parse arguments
    image_size = cp["TRAIN"].getint("image_size")
    alpha = cp["TRAIN"].getfloat("alpha")
    use_trained_model = cp["TRAIN"].getboolean("use_trained_model")
    source_folder = cp["TRAIN"].get("source_folder")
    target_folder = cp["TRAIN"].get("target_folder")
    epochs = cp["TRAIN"].getint("epochs")
    train_steps = cp["TRAIN"].getint("train_steps")
    learning_rate = cp["TRAIN"].getfloat("learning_rate")
    epochs_to_reduce_lr = cp["TRAIN"].getint("epochs_to_reduce_lr")
    reduce_lr = cp["TRAIN"].getfloat("reduce_lr")
    output_model = cp["TRAIN"].get("output_model")
    output_log = cp["TRAIN"].get("output_log")
    batch_size = cp["TRAIN"].getint("batch_size")
    verbose = cp["TRAIN"].getboolean("verbose")

    # Training
    trained_model = None
    if use_trained_model:
        trained_model = cp["TRAIN"].get("trained_model")
    model = AELikeModel(image_size, alpha,verbose, trained_model)
    model.train(source_folder, target_folder, epochs, train_steps, learning_rate, epochs_to_reduce_lr, reduce_lr, output_model, output_log, batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hmchuong - BoneSuppression v2 - Training')
    parser.add_argument('--config', default='config/train.cfg', type=str, help='config file')
    args = parser.parse_args()
    main(args)

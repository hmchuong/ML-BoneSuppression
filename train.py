from configparser import ConfigParser
from preprocessing import ImageProcessing
from model import AELikeModel

def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # Parse arguments
    is_registration = cp["DATA"].getboolean("image_registration")
    registered_images_dir = cp["DATA"].get("registered_images_dir")
    image_size = cp["DATA"].getint("image_size")
    is_augment_data = cp["DATA"].getboolean("augment_data")
    alpha = cp["TRAIN"].getfloat("alpha")
    use_trained_model = cp["TRAIN"].getboolean("use_trained_model")
    x_path_pattern = cp["TRAIN"].get("x_pattern")
    y_path_pattern = cp["TRAIN"].get("y_pattern")
    queue_capacity = cp["TRAIN"].getint("queue_capacity")
    capacity = cp["TRAIN"].getint("capacity")
    min_after_dequeue = cp["TRAIN"].getint("min_after_dequeue")
    num_threads = cp["TRAIN"].getint("num_threads")
    x_test_dir = cp["TRAIN"].get("x_test_dir")
    y_test_dir = cp["TRAIN"].get("y_test_dir")
    epochs = cp["TRAIN"].getint("epochs")
    train_steps = cp["TRAIN"].getint("train_steps")
    learning_rate = cp["TRAIN"].getfloat("learning_rate")
    epochs_to_reduce_lr = cp["TRAIN"].getint("epochs_to_reduce_lr")
    reduce_lr = cp["TRAIN"].getfloat("reduce_lr")
    output_dir = cp["TRAIN"].get("output_dir")
    batch_size = cp["TRAIN"].getint("batch_size")

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
    trained_model = None
    if use_trained_model:
        trained_model = cp["TRAIN"].get("trained_model")
    model = AELikeModel(image_size, alpha, trained_model)
    model.train(x_path_pattern, y_path_pattern, queue_capacity, capacity, min_after_dequeue, num_threads, x_test_dir, y_test_dir, epochs, train_steps, learning_rate, epochs_to_reduce_lr, reduce_lr, output_dir, batch_size)

if __name__ == '__main__':
    main()

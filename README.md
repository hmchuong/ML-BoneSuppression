# Bone Suppression from Chest Radiographs

The project is a tool to build **Bone Suppression** model, written in tensorflow

<img src="description.png" alt="CAM example image"/>

## What is [Bone Suppression](https://www.researchgate.net/publication/320252756_Deep_learning_models_for_bone_suppression_in_chest_radiographs?enrichId=rgreq-7b19be48d9763ea61b22252eaf96edca-XXX&enrichSource=Y292ZXJQYWdlOzMyMDI1Mjc1NjtBUzo1ODQ1MzY0NDY0ODAzODRAMTUxNjM3NTc1NzU5Nw%3D%3D&el=1_x_3&_esc=publicationCoverPdf)?
Bone suppression is an autoencoder-like model for eliminating bone shadow from Chest X-ray images. The model require two types of dataset: normal  and bone-suppression X-ray images. The target model can suppress bone shadow from Chest X-ray images, help Radiologists diagnose better lung related diseases. Although there are some softwares supporting bone suppression ([ClearRead](https://www.riveraintech.com/clearread-xray/), [CareStream](https://www.itnonline.com/content/carestream%E2%80%99s-new-bone-suppression-software-receives-fda-clearance)), this project is a practical open source in computer vision and deep learning.

## In this project you can
1. Preprocessing data, including registration and augmentation.
2. Train/test by following the quickstart. You can get a model with performance close to the paper.
3. Visualize your training result with tensorboard

## Requirements
The project requires `Python>=3.5`.

I have trained on an instance with `1 NVIDIA GTX 1080Ti (11GB VRAM)` and it takes approximately 14 hours.

## Configuration
### [DATA](config/data_preprocessing.cfg)
1. You can download the dataset [here](https://www.kaggle.com/hmchuong/xray-bone-shadow-supression). This dataset includes 3 parts: `JSRT` dataset in `png` format, `BSE_JSRT` dataset in `png` format, and `augmented` dataset which can be trained directly.
2. To register the dataset, make sure you set `data_registration` to `true`, and the input images are read from `source_dir` (JSRT) and `target_dir` (BSE_JSRT). The registered images will be saved to `registered_output_dir` into `source` and `target` subdirectories.
3. To augment the dataset, make sure you set `data_augmentation` to `true`, the `source_dir` and `target_dir` will be used to augment. `The total data after augmentation for source/target` = `augmentation_seed` X total number of images in `source_dir` or `target_dir`. The augmented images will be saved to `source` and `target` subdirectories of `augmented_output_dir` with `.png` extension.

### [TRAIN](config/train.cfg)
1. `source_folder` and `target_folder` are folders to load training images.
4. If you want to continue training from your last model, set `use_trained_model` to true and `trained_model` to your model path.
5. `output_model` is where you save your model during training and `output_log` is where you save the tensorboard checkpoints.
6. The other parameters is set following the published paper

## Pretrained model
If you want to start testing without training from scratch, you can use the [model](/model) I have trained. The model has loss value: 0.01409, MSE: 7.1687e-4, MS-SSIM: 0.01517

## Quickstart
**Note that currently this project can only be executed in Linux and macOS. You might run into some issues in Windows.**
1. Create & activate a new python3 virtualenv. (optional)
2. Install dependencies by running `pip install -r requirements.txt`.
3. Run `python train.py` to train a new model. If you want to change your config path:
```
python train.py --config <config path>
```
During training, you can use Tensorboard to visualize the results:
```
tensorboard --logdir=<output_log in train.cfg>
```
4. Run `python test.py` to evaluate your model on specific image. To change default parameters, you can use:
```
python test.py --model <model_path> --config <model config path> --input <image path> --output <output image path>
```

## Acknowledgement
I would like to thank [LoudeNOUGH](https://github.com/LoudeNOUGH/bone-suppression) for scratch training script and Hussam Habbreeh (حسام هب الريح) for sharing his experiences on this task.

## Author
Chuong M. Huynh (minhchuong.itus@gmail.com)

## License
MIT

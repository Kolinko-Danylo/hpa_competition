import numpy as np
# import nibabel as nib
import os
import glob
import yaml
import tqdm
import shutil
from hpa_competition.definitions import PREPROCESSING_CONFIG_PATH

CHANNELS = ["flair", "t1", "t1ce", "t2"]
SEGMENTATION = "seg"


def create_output_dirs(path):
    """ Create directories for preprocessed images """

    def _create_output_dirs(child_path, folder_names):
        """ Create a directory with child folders """
        if os.path.exists(child_path):
            shutil.rmtree(child_path)

        os.makedirs(child_path)
        for folder_name in folder_names:
            os.makedirs(os.path.join(child_path, folder_name))

    train_3d_path = os.path.join(path, "train_3d")
    val_3d_path = os.path.join(path, "val_3d")
    train_2d_path = os.path.join(path, "train_2d")
    val_2d_path = os.path.join(path, "val_2d")

    _create_output_dirs(train_3d_path, folder_names=[])
    _create_output_dirs(val_3d_path, folder_names=[])
    _create_output_dirs(train_2d_path, folder_names=["X", "y"])
    _create_output_dirs(val_2d_path, folder_names=["X", "y"])

    return train_3d_path, val_3d_path, train_2d_path, val_2d_path


def split_data(path, validation_size):
    """ Split data into train and validation parts """

    images_paths = glob.glob(os.path.join(path, "?GG", "*"))
    images_paths.sort()

    np.random.seed(42)
    np.random.shuffle(images_paths)

    train_instance_number = int(len(images_paths) * (1 - validation_size))
    return images_paths[:train_instance_number], images_paths[train_instance_number:]


def fill_3d_data_folders(sources, destination):
    """ Fill destination folder with 3d images """

    for source in tqdm.tqdm(sources, desc="[INFO] Copying train/val 3d images"):
        destination_folder = os.path.join(destination, os.path.basename(source))
        os.mkdir(destination_folder)

        for file_name in os.listdir(source):
            shutil.copyfile(os.path.join(source, file_name), os.path.join(destination_folder, file_name))


def read_instance(path, shape=(4, 240, 240, 155)):
    """ Read a 3-d image with corresponding mask """

    x = np.empty(shape)
    for i, channel in enumerate(CHANNELS):
        channel_path = glob.glob(os.path.join(path, f"*{channel}.nii.gz"))[0]
        x[i] = nib.load(channel_path).get_fdata()

    y_path = glob.glob(os.path.join(path, f"*{SEGMENTATION}.nii.gz"))[0]
    y = nib.load(y_path).get_fdata()
    return x, y


def fill_2d_data_folders(sources, destination, threshold=None):
    """ Fill destination folder with 2-d image slices """

    def _save_slice(x_slice, y_slice, path, i):
        """ Save single slice of image """
        np.save(os.path.join(path, "X", f"{i}.npy"), x_slice.astype(np.uint16))
        np.save(os.path.join(path, "y", f"{i}.npy"), y_slice.astype(np.uint8))

    counter = 0
    for source in tqdm.tqdm(sources, desc="[INFO] Copying train/val 2d images"):
        x, y = read_instance(source)

        for i in range(y.shape[-1]):
            x_slice, y_slice = np.transpose(x[:, :, :, i], (1, 2, 0)), y[:, :, i]

            if threshold is None or (y_slice != 0).mean() > threshold:
                _save_slice(x_slice, y_slice, destination, counter)
                counter += 1


if __name__ == "__main__":
    with open(PREPROCESSING_CONFIG_PATH, "r") as config_file:
        config = yaml.full_load(config_file)

    train_3d_path, val_3d_path, train_2d_path, val_2d_path = create_output_dirs(config["preprocessed_path"])
    train_instance_paths, val_instance_paths = split_data(config["train_input_path"], config["validation_size"])

    fill_3d_data_folders(train_instance_paths, train_3d_path)
    fill_3d_data_folders(val_instance_paths, val_3d_path)

    fill_2d_data_folders(train_instance_paths, train_2d_path, threshold=config["background_threshold"])
    fill_2d_data_folders(val_instance_paths, val_2d_path)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def getImagePaths(data_path):

    masks_paths = []

    for directory in os.listdir(data_path):
        # excluding readme and data.csv files as they are not used in training
        if directory not in ['README.md', 'data.csv']:
            for file_path in os.listdir(os.path.join(data_path, directory)):
                if '_mask' in file_path:
                    masks_paths.append(os.path.join(data_path, directory, file_path))

    # replacing '_mask' in masks path to get image paths
    image_paths = [path.replace("_mask", '') for path in masks_paths]

    print("Total number of images found: ", len(image_paths))
    print("Total number of masks found: ", len(image_paths))

    # converting to dataframe as flow_from_dataframe requires dataframe as input
    return pd.DataFrame({"images": image_paths}), pd.DataFrame({"masks": masks_paths})


def splitData(paths_of_images, paths_of_masks, test_size=0.2, random_state=42):
    # splitting the data into train and validation sets
    return train_test_split(paths_of_images, paths_of_masks, test_size=test_size, random_state=random_state)


if '__name__' == '__main__':

    PATH = "/home/sai/Desktop/BRATS/kaggle_3m"

    all_image_paths, all_mask_paths = getImagePaths(PATH)

    # splitting the train set into train, validation and test sets
    train_images, test_images, train_masks, test_masks = splitData(all_image_paths, all_mask_paths, test_size=0.1)
    train_images, val_images, train_masks, val_masks = splitData(train_images, train_masks, test_size=0.2)

    # dictionary to store all extracted paths
    make_dataset = {"train_images": train_images,
                    "train_masks": train_masks,
                    "val_images": val_images,
                    "val_masks": val_masks,
                    "test_images": test_images,
                    "test_masks": test_masks}

    # saving the train, validation and test files absolute paths as a dictionary
    np.save(os.path.join(PATH, 'data', 'CT_dataset'), make_dataset)

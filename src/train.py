import numpy as np
from data_generator import dataGenCreator
from experiment import Train
from model import UNetModel

if '__name__' == '__main__':
    INPUT_SIZE = (256, 256, 3)
    TARGET_SIZE = (256, 256)
    IMG_CHANNELS = 144
    N_FILTERS = 64
    DROPOUT = 0.1
    BATCHNORM = True
    POOLING_SIZE = 2
    KERNEL_SIZE = 3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 5

    PATH = ""
    MODEL_SAVE_PATH = os.path.join(PATH, 'models')
    dataset_path = ''  # path for dictory .npy wher eall paths are stored using data_load.py

    # we create two instances with the same arguments
    data_gen_args = dict(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)

    dataset_path = np.load(dataset_path, allow_pickle=True).tolist()

    print("Train data generator:-")
    train_data_generator = dataGenCreator(dataset_path["train_images"], dataset_path["train_masks"], data_gen_args,
                                          "images", "masks", BATCH_SIZE, TARGET_SIZE)
    print("Validation data generator:-")
    validation_data_generator = dataGenCreator(dataset_path["val_images"], dataset_path["val_masks"], data_gen_args,
                                               "images", "masks", BATCH_SIZE, TARGET_SIZE)
    print("Test data generator:-")
    test_data_generator = dataGenCreator(dataset_path["test_images"], dataset_path["test_masks"], data_gen_args,
                                         "images", "masks", BATCH_SIZE, TARGET_SIZE)

    # create a unet model with required hyper parameters
    unetModel = UNetModel(INPUT_SIZE, n_filters=N_FILTERS, dropout=DROPOUT,
                          batchnorm=BATCHNORM)

    unet = UNetModel(input_size=INPUT_SIZE, n_filters=N_FILTERS, dropout=DROPOUT, batchnorm=BATCHNORM,
                     pooling_size=POOLING_SIZE, kernel_size=KERNEL_SIZE)
    unetModel = unet.get_unet()

    trainObj = Train(model=unetModel, learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     model_dest_path=MODEL_SAVE_PATH, train_data_generator=train_data_generator,
                     len_train_images=len(dataset_path["train_images"]), val_data_generator=validation_data_generator,
                     len_val_images=len(dataset_path["val_images"]), smooth=100)

    trainObj.train()

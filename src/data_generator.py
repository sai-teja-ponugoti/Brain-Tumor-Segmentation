from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dataGenCreator(images, masks, args_dict, image_col, paths_col, batch_size, target_size):
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_datagen = ImageDataGenerator(rescale=1 / 255, **args_dict)
    mask_datagen = ImageDataGenerator(rescale=1 / 255, **args_dict)

    image_generator = image_datagen.flow_from_dataframe(
        images,
        x_col=image_col,
        batch_size=batch_size,
        target_size=target_size,
        color_mode="rgb",
        class_mode=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        masks,
        x_col=paths_col,
        batch_size=batch_size,
        target_size=target_size,
        color_mode="grayscale",
        class_mode=None,
        seed=seed)

    # combine generators into one which yields image and masks
    return zip(image_generator, mask_generator)


from tensorflow.keras.preprocessing.image import ImageDataGenerator


def augment_images(folder,
                   batch_size=128,
                   output_shape=(150, 150),
                   rotation_range=45,
                   width_shift_range=.1,
                   height_shift_range=.1,
                   zoom_range=.1,
                   horizontal_flip=True,
                   vertical_flip=True,
                   rescale=1. / 255,
                   ):
    image_gen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale)

    data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                             directory=folder,
                                             shuffle=True,
                                             target_size=output_shape,
                                             class_mode='sparse')

    return data_gen

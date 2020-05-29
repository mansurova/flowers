import logging
import os
import shutil
from glob import glob

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO,
                    format='%(name)s\t[%(levelname)-8s] %(message)s')
logger = logging.getLogger('Augment Data')

logger.info('Tensorflow Version: %s' % tf.__version__)


# Load images
cwd = os.getcwd()
base_dir = os.path.join(cwd, 'img')

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
for cl in classes:
    img_path = os.path.join(base_dir, 'original-jpegs', cl)
    logger.info('img_path: %s', img_path)
    images = glob(img_path + '/*.jpg')
    logger.info("%s:\t%s Images" % (cl, len(images)))
    train, val = train_test_split(images, test_size=.2)

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))
        shutil.copy(t, os.path.join(base_dir, 'train', cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
        shutil.copy(v, os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')


batch_size = 128
epochs = 15
img_shape = 150, 150

image_gen = ImageDataGenerator(# featurewise_center=True,
                               # featurewise_std_normalization=True,
                               rotation_range=90,
                               width_shift_range=.2,
                               height_shift_range=.2,
                               zoom_range=.2,
                               horizontal_flip=True,
                               vertical_flip=True,
                               rescale=1./255)

train_data_gen =  image_gen.flow_from_directory(batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=img_shape,
                                                class_mode='binary')

# Create images and save into folder output
for i in range(epochs):
    batch_image, batch_type = next(train_data_gen)
    for j in range(batch_size):
        image, type = batch_image[j,:,:,:], batch_type[j]
        dirname = os.path.join(cwd, 'output', classes[int(type)])
        filename = os.path.join(dirname, '%s_%s.jpg' % (i,j))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.imsave(filename, image)

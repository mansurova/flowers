import os
import shutil
from glob import glob

from sklearn.model_selection import train_test_split


# # Data Preprocessing
# Execute this Code only once because it will move the data into a new training and validation directory:

# In[4]:

# Define classes
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load images
cwd = os.getcwd() # current working directory
base_dir = os.path.join(cwd[:-7], 'data') # base image directory
for cl in classes:
    img_path = os.path.join(base_dir, 'flowers', 'train', cl)
    images = glob(img_path + '/*.jpg')

    # Split images into train an validation
    train, val = train_test_split(images, test_size=.2,
                                  shuffle=True, random_state=42)

    # Move images into training directory
    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))
        shutil.move(t, os.path.join(base_dir, 'train', cl))

    # Move images into validation directory
    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
        shutil.move(v, os.path.join(base_dir, 'val', cl))

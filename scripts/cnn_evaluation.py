#!/usr/bin/env python
# coding: utf-8

# In[109]:


import logging
import os

import numpy as np
import tensorflow as tf
from sklearn import metrics
from augment_data import augment_images

import matplotlib.pyplot as plt

# Logging Configuration
logging.basicConfig(level=logging.INFO,
                    format='%(name)s\t[%(levelname)-8s] %(message)s')
logger = logging.getLogger('CNN Eval')
logger.info('Tensorflow Version: %s' % tf.__version__)


# # Loading Data
# ## Loading pre-trained model

# Using 2 pre trained models `model` and `model2` to analyse.
# 
# `cnn_model_mobilenet` usied the pretrained [Mobilenet](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) and trained this with Google Colab to our dataset.
# 
# `cnn_model_resnet` usied the pretrained [Resnet](https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4) and trained this with Google Colab to our dataset.

# In[20]:


# Loading trained model
model_name = '../model/cnn_model_mobilenet'
model = tf.keras.models.load_model(model_name)
model_name = '../model/cnn_model_resnet'
model2 = tf.keras.models.load_model(model_name)


# ## Loading data

# In[15]:


cwd = os.getcwd() # current working directory
base_dir = os.path.join(os.path.split(cwd)[0], 'data')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'flowers', 'test')


# In[16]:


# Define classes
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


# # Visualizing Predictions

# In[19]:


batch_size = 30
img_height, img_width = 224, 224
train_image_gen = augment_images(train_dir, 
        batch_size=batch_size,
        output_shape=(img_height, img_width),
        )
val_image_gen = augment_images(val_dir, 
        batch_size=batch_size,
        output_shape=(img_height, img_width),
        )
image_batch, label_batch = next(val_image_gen)

predictions = model.predict(image_batch)
predicted_ids = np.argmax(predictions, axis=-1)
plt.figure(figsize=(16,15))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.subplots_adjust(hspace = 0.3)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(classes[predicted_ids[n]].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
plt.savefig("../output/cnn_model_predictions3.png")


# In[115]:


FIGSIZE = 16, 10
def get_predictions(image_gen, model):
    image_batch, y_true = next(image_gen)
    y_pred = model.predict(image_batch)

    return y_pred, y_true

def get_metrics(y_true, y_pred):
    predicted_ids = np.argmax(y_pred, axis=-1)
    metric = dict(
        conf_mx = metrics.confusion_matrix(y_true, predicted_ids),
        multi_conf_mx = metrics.multilabel_confusion_matrix(y_true, predicted_ids),
        f1 = metrics.f1_score(np.int64(y_true), predicted_ids, average=None),
        acc = metrics.accuracy_score(y_true, predicted_ids),
        avg_recall = metrics.recall_score(y_true, predicted_ids, average=None),
        avg_precision = metrics.precision_score(y_true, predicted_ids, average=None),
        report = metrics.classification_report(y_true, predicted_ids, 
                                               labels=[i for i in range(5)], 
                                               target_names=classes)
    )
    return metric

def plot_conf_mx(conf_mx, title):
    fig = plt.figure(figsize=FIGSIZE)
    plt.subplot(1,2,1)
    plt.imshow(conf_mx)
    plt.colorbar()
    plt.title("Confusion Matrix %s" % title)

    plt.subplot(1,2,2)
    np.fill_diagonal(conf_mx, 0)
    plt.imshow(conf_mx)
    plt.colorbar()
    plt.title("Confusion Matrix %s (without diagonals)" % title)
    plt.savefig('../output/confusion_matrix_%s.png' % title)
    plt.show()

def plot_f1(f1, title):
    plt.figure(figsize=FIGSIZE)
    plt.bar(classes, f1)
    plt.ylim([0,1])
    plt.title("F1 Score %s" % title)
    plt.savefig('../output/f1_score_%s.png' % title)
    plt.show()

def plot_precision_recall(label_batch, y_pred, title):
    predicted_ids = np.argmax(y_pred, axis=-1)
    avg_prec = metrics.precision_score(label_batch, predicted_ids, average=None)
    fig, ax = plt.subplots(figsize=(16,9))
    for i, cl in enumerate(classes):
        y_true = label_batch == i
        probas_pred = y_pred[:,i]
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, probas_pred)
        metrics.PrecisionRecallDisplay(precision, recall, 
                                       average_precision=avg_prec[i], 
                                       estimator_name=None).plot(ax=ax, name=cl)
    plt.savefig('../output/precision_recall_%s.png' % title)


# In[116]:


img_size = 224, 224
train_image_gen = augment_images(train_dir, 
        batch_size=4030,
        output_shape=img_size,
        )
val_image_gen = augment_images(val_dir, 
        batch_size=1010,
        output_shape=img_size,
        )
test_image_gen = augment_images(test_dir, 
        batch_size=1260,
        output_shape=img_size,
        )

predictions = {}
metric = {}

for mname, m in [("mobilenet", model), 
                 ("resnet", model2)]:
    for title, image_gen in [('%s_Training' % mname, train_image_gen),
                             ('%s_Validation' % mname, val_image_gen),
                             ('%s_Test' % mname, test_image_gen),
                            ]:
        # Get predictions
        logger.info('Get Predictions for %s' % title)
        y_pred, y_true = get_predictions(image_gen, m)
        predictions[title] = {'y_pred': y_pred,
                              'y_true': y_true}

        # Calculate metrics
        logger.info('Calculate Metrics for %s' % title)
        metric[title] = get_metrics(y_true, y_pred)

        # Print important informations
        print(metric[title].get('report'))

        # Plot important metrics
        logger.info('Plotting for %s' % title)
        plot_conf_mx(metric[title].get('conf_mx'), title)
        plot_f1(metric[title].get('f1'), title)
        plot_precision_recall(y_true, y_pred, title)


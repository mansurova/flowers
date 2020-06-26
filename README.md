# Flower Projekt
Implementation vom Flower Projekt für die Data Science 1 Veranstaltung.

This code is traning two models to classify an image into one of the following groups:
* daisy
* dandelion
* rose
* sunflower
* tulip

# Requirements to run the code
To run this code you need to download the images from the following Datasets:
1. https://www.kaggle.com/mgornergoogle/five-flowers
2. https://www.kaggle.com/ianmoone0617/flower-goggle-tpu-classification

and put them into the folder `data/data1` (five-flowers) and `data/data2`.

Important: The folder structure must appear as follows:
* `data/` [data1, data2]
* `flowers/`

Otherwise you will not be able to re-run the code.
It is recommended to run the code in a virtual environment (e.g. `virtualenv`). `requirements.txt` contains all the needed Python libraries.

Alternatively, it is already done in the docker container [`ody55eus/flowers`](https://hub.docker.com/repository/docker/ody55eus/flowers/).


# Data preparation

The following preprocessing steps are performed by [`data_preparation.py`](scripts/data_preparation.py):
* Copy both datasets together .
  * Delete all pictures that are not fitting into one of the given 5 groups.
* Identify and delete duplicate images.
* Split the dataset into train (80%) and test (20%) data.



# Convolutional Neural Network (CNN)

The convolutional neural network (CNN) training can be reconstructed with the file [`cnn_training.py`](scripts/cnn_training.py)


Before training the CNN the training data are split again into train (80%) and validation (20%) data (this is performed by [`cnn_training.py`](scripts/cnn_training.py) or [`cnn_split_test_val.py`](scripts/cnn_split_test_val.py)).
If you choose a higher number of `epochs`, you can probably improve the result.

## Training the model

The training was processed with [Google Colab](https://drive.google.com/file/d/1xMJ1Kt4YBeIpqGIzPt1Km8ziwNW5a2Og/view?usp=sharing) to take advantage of the fast computation with GPU.

## Evaluation 

The Evaluation of the CNN models can be reproduced by running [`evaluation.py`](scripts/evaluation.py).

# Support Vector Machines

The support vector machines training and evaluation can be reproduced by [`svm.ipynb`](scripts/svm.ipynb)


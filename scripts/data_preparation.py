"""
Author: Diana Mansurova (5786387)
Inspirared by:
https://www.kaggle.com/deepcnn/mislabeled-training-images 
https://www.kaggle.com/rdokov/exact-duplicates 
"""
import numpy as np
from sklearn.model_selection import train_test_split
import hashlib 
import cv2
import os
import glob
import shutil
import matplotlib.pyplot as plt
import pandas as pd  
import time 

def extract_data2():
    """
    Extracts images that contain five classes of interest
    Renames images to the following format:
        {class}2_{images_number}.jpeg
    Moves them to a corresponding class folder: flowers/{class}.
    Return:
        flowers - a dictionary with five keys ( 1 sunflower, 2 tulip, 3 rose, 4 dandelion, 5 daisy)
    """

    path_train = "../data/data2/flower_tpu/flower_tpu/flowers_google/flowers_google/"

    imgs = sorted(glob.glob(path_train + "*.jpeg"))
    labels = pd.read_csv("../data/data2/flowers_idx.csv")

    match = {
        'sunflower': 1,
        'common tulip': 2,
        'siam tulip': 2,
        'wild rose': 3,
        'rose': 3,
        'desert-rose': 3,
        'lenten rose': 3,
        'pink primrose': 3,
        'common dandelion': 4,
        'daisy': 5,
        'barberton daisy': 5
    }

    # descripton of the folder numbering
    folder = {
        1: "sunflower",
        2: "tulip",
        3: "rose",
        4: "dandelion",
        5: "daisy"
    }

    data = labels.loc[labels['flower_cls'].isin(list(match.keys()))]
    
    # data.to_csv('data.csv', index=False)
    ids = data['id'].values

    for img in imgs:
        num = int(img[len(path_train):-5])
        if num in ids:
            cl = match[data.loc[data['id'] == num]["flower_cls"].values[0]]
            os.rename(img,  "../data/flowers/" +
                            folder[cl] + 
                            "/" + 
                            folder[cl] + 
                            "2" + "_" + 
                            str(num) + ".jpeg")
    

def extract_data1():
    """
    Renames images to the following format:
        {class}1_{images_number}.jpeg
    Moves them to a corresponding class folder: data/flowers/{class}.
    """
    path = "../data/data1/original-jpegs/"
   
    classes = sorted(glob.glob(path + "*"))
    IDX = 1
    for cl in classes:
        imgs = sorted(glob.glob(cl + "/*.jpg"))
        cl_name = cl[len(path):]
        if cl_name[-1] == "s":
            cl_name = cl_name[:-1]
        for img in imgs:
            os.rename(img,  "../data/flowers/" +
                                cl_name + 
                                "/" + 
                                cl_name + 
                                "1" + "_" + 
                                str(IDX) + ".jpg")
            IDX += 1


def merge_data():
    """
    Merges two datasets and stores them in ../data/flowers/
    """
    # create new folder structure
    if not os.path.exists("../data/flowers/"):
        os.mkdir("../data/flowers/")
    if not os.path.exists("../data/flowers/dandelion/"):
        os.mkdir("../data/flowers/dandelion/")
    if not os.path.exists("../data/flowers/rose/"):
        os.mkdir("../data/flowers/rose/")
    if not os.path.exists("../data/flowers/sunflower/"):
        os.mkdir("../data/flowers/sunflower/")
    if not os.path.exists("../data/flowers/daisy/"):
        os.mkdir("../data/flowers/daisy/")
    if not os.path.exists("../data/flowers/tulip/"):
        os.mkdir("../data/flowers/tulip/")

    # move the License file to the flower/ folder
    if os.path.exists("../data/data1/LICENSE.txt") :
        os.rename("../data/data1/LICENSE.txt", "../data/flowers/LICENSE.txt")

    # Extract data
    extract_data1()
    extract_data2()

    # Delete remaining old folders
    if os.path.exists("../data/data1/"):
        shutil.rmtree("../data/data1/")
    
    if os.path.exists("../data/data2/"):
        shutil.rmtree("../data/data2/")

merge_data()

def remove_exact_duplicates():
    """
    removes the completely identical images using the md5sum hash
    Adapted from https://www.kaggle.com/rdokov/exact-duplicates  
    """
    records = []

    path = '../data/flowers/'
    dirs =  glob.glob(path + "*/")
    for cl in dirs:
        cl = cl[len(path):-1]
        for img in  os.listdir(path + cl):
            
            with open(path + cl + "/" + img, 'rb') as fd:
                md5sum = hashlib.md5(fd.read()).hexdigest()

            records.append({
                'filename': img,
                'class': cl,
                'md5sum': md5sum,
            })

    df = pd.DataFrame.from_records(records)


    counts = df.groupby('md5sum')['class'].count()
    duplicates = counts[counts > 1]
    print("Number of exact duplicates: ", len(duplicates))

    # print(duplicates)
    for md5sum in duplicates.index:
        subset = df[df['md5sum'] == md5sum]
        print(subset)
        if len(subset['filename'].value_counts()) > 1:
            
            img1_name = path + subset.iloc[0, 1] + "/" + subset.iloc[0, 0]
            img2_name = path + subset.iloc[1, 1] + "/" + subset.iloc[1, 0]

            # visualize duplicates
            img1 = cv2.cvtColor(cv2.imread(img1_name), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(img2_name), cv2.COLOR_BGR2RGB)
            
            fig = plt.figure()
            fig.add_subplot(121)
            plt.title(subset.iloc[0, 0])
            plt.imshow(img1)

            fig.add_subplot(122)
            plt.title(subset.iloc[1, 0])
            plt.imshow(img2)

            plt.show()
            
            if os.path.exists(img1_name):
                os.remove(img1_name)
            if os.path.exists(img2_name):
                os.remove(img2_name)

            print('------')

def crop_to_square(img, resolution=256):
    """
    crops and resizes an image to a certain square size e.g. 256 x 256
    """
    h, w = img.shape[:2]

    left = 0
    right = w
    top = 0
    bottom = h 

    half_w = w // 2
    half_h = h // 2

    if h > w:
        left = 0
        right = w
        top = half_h - half_w
        bottom = half_h + half_w

        if bottom - top > right:
            bottom -=  bottom - top - right
        elif bottom - top < right:
            bottom +=  right - (bottom - top)

    elif w > h:
        left = half_w - half_h
        right = half_w + half_h
        top = 0
        bottom = h 

        if right - left > bottom:
            right -=  right - left - bottom
        elif right - left < bottom:
            right += bottom - (right - left)

    img = cv2.resize(img[top:bottom, left:right], (resolution, resolution))
    return img


def load_data(resolution):
    """
    Loads the entire dataset in a list
    """
    path = "../data/flowers/"
    clss = glob.glob(path + "*/")
    
    images = []
    names = []
    classes = []

    for cl in clss:
        imgs = glob.glob(cl + "*.j*")
        for img in imgs:
            classes += [cl[len(path):-1]]
            images += [crop_to_square(cv2.imread(img, 0), resolution=resolution)]
            names += [img[len(cl):]]
    return images, names, classes


# The function below will find any
# image similar to the input image

def mse(img1, img2):
    err = np.sum((img1 - img2) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def find_pairs(compare_img, compare_idx, images, names, matches):
    """ 
    Find all paired duplicates
    Adapted from https://www.kaggle.com/deepcnn/mislabeled-training-images 
    """
    threshold = 90 # less than 10% difference
    threshold = 10000
    for idx, img in enumerate(images):
        if idx <= compare_idx:
            continue
        else:   
            if np.abs(compare_img - img).sum() < threshold \
                and idx != compare_idx:
                matches.append((names[compare_idx], names[idx])) #(1 - mse(compare_img, img))*100 >= threshold \
    return matches


def remove_duplicates(matches, visualize=True):
    for (path1, path2) in matches:
        if "tulip" in path1:
            img1 = "../data/flowers/tulip/" + path1
        elif "sunflower" in path1:
            img1 = "../data/flowers/sunflower/" + path1
        elif "daisy" in path1:
            img1 = "../data/flowers/daisy/" + path1
        elif "rose" in path1:
            img1 = "../data/flowers/rose/" + path1
        else:
            img1 = "../data/flowers/dandelion/" + path1

        if "tulip" in path2:
            img2 = "../data/flowers/tulip/" + path2
        elif "sunflower" in path2:
            img2 = "../data/flowers/sunflower/" + path2
        elif "daisy" in path2:
            img2 = "../data/flowers/daisy/" + path2
        elif "rose" in path2:
            img2 = "../data/flowers/rose/" + path2
        else:
            img2 = "../data/flowers/dandelion/" + path2

        if visualize:
            im1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
            im2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
            fig = plt.figure()
            fig.add_subplot(121)
            plt.title(path1)
            plt.imshow(im1)

            fig.add_subplot(122)
            plt.title(path2)
            plt.imshow(im2)
            plt.show()
        if os.path.exists(img1):
            os.remove(img1)
        if os.path.exists(img2):
            os.remove(img2)


remove_exact_duplicates()
images, names, classes = load_data(resolution=100)


"""
# find all the duplicates
t0 = time.time()
matches = []
for idx, img in enumerate(images):
    matches = find_pairs(img, idx, images, names, matches)   
    if idx % 100 == 0:
        print(idx)
        print(matches)
print(len(matches))
print(matches)
print("Time elapsed: ", time.time() - t0)

# Prepare splitting of the dataset into training and testing sets with respect to the ratios 80:20
remove_duplicates(matches, visualize=True)
"""

def create_new_folder_structure():
    # create new folder structure
    if not os.path.exists("../data/flowers/train/"):
        os.mkdir("../data/flowers/train/")
    if not os.path.exists("../data/flowers/train/dandelion/"):
        os.mkdir("../data/flowers/train/dandelion/")
    if not os.path.exists("../data/flowers/train/rose/"):
        os.mkdir("../data/flowers/train/rose/")
    if not os.path.exists("../data/flowers/train/sunflower/"):
        os.mkdir("../data/flowers/train/sunflower/")
    if not os.path.exists("../data/flowers/train/daisy/"):
        os.mkdir("../data/flowers/train/daisy/")
    if not os.path.exists("../data/flowers/train/tulip/"):
        os.mkdir("../data/flowers/train/tulip/")

    if not os.path.exists("../data/flowers/test/"):
        os.mkdir("../data/flowers/test/")
    if not os.path.exists("../data/flowers/test/dandelion/"):
        os.mkdir("../data/flowers/test/dandelion/")
    if not os.path.exists("../data/flowers/test/rose/"):
        os.mkdir("../data/flowers/test/rose/")
    if not os.path.exists("../data/flowers/test/sunflower/"):
        os.mkdir("../data/flowers/test/sunflower/")
    if not os.path.exists("../data/flowers/test/daisy/"):
        os.mkdir("../data/flowers/test/daisy/")
    if not os.path.exists("../data/flowers/test/tulip/"):
        os.mkdir("../data/flowers/test/tulip/")


def split_dataset():
    """ 
    Slpit the entire dataset into training and testing set
    """
    create_new_folder_structure()
    path = "../data/flowers/"
    tulip = glob.glob(path + "tulip/*.j*")
    sunflower = glob.glob(path + "sunflower/*.j*")
    rose = glob.glob(path + "rose/*.j*")
    dandelion = glob.glob(path + "dandelion/*.j*")
    daisy = glob.glob(path + "daisy/*.j*")
    flowers = [tulip, daisy, sunflower, rose, dandelion]

    minimum_size= min([len(daisy), len(dandelion), len(rose), len(sunflower), len(tulip) ])
    
    for i in range(0,3):
        for fl in flowers:
            np.random.seed(42)
            np.random.shuffle(fl)

    for idx, fl in enumerate(flowers):
        fl = fl[:minimum_size]
        X_train, X_test = train_test_split(fl, test_size=0.2, shuffle=True, random_state=42)
        
        # relocate the training set
        for sample in X_train:
            img = crop_to_square(cv2.imread(sample), resolution=256)
            cv2.imwrite("../data/flowers/train/" + sample[len(path):], img)
        # relocate the testing set
        for sample in X_test:
            img = crop_to_square(cv2.imread(sample), resolution=256)
            cv2.imwrite("../data/flowers/test/" + sample[len(path):], img)    
    

    if os.path.exists("../data/flowers/daisy/"):
        shutil.rmtree("../data/flowers/daisy/")
    if os.path.exists("../data/flowers/sunflower/"):
        shutil.rmtree("../data/flowers/sunflower/")
    if os.path.exists("../data/flowers/rose/"):
        shutil.rmtree("../data/flowers/rose/")
    if os.path.exists("../data/flowers/tulip/"):
        shutil.rmtree("../data/flowers/tulip/")
    if os.path.exists("../data/flowers/dandelion/"):
        shutil.rmtree("../data/flowers/dandelion/")

split_dataset()
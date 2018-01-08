from __future__ import division, print_function, absolute_import
from skimage import color, io
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split
import os
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
import numpy as np
import cv2



###################################
### Import picture files
###################################

files_path = 'D://train1'

cat_files_path = os.path.join(files_path, 'cat*.jpg')
dog_files_path = os.path.join(files_path, 'dog*.jpg')

cat_files = sorted(glob(cat_files_path))
dog_files = sorted(glob(dog_files_path))

n_files = len(cat_files) + len(dog_files)
print(n_files)

size_image = 64

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0
for f in cat_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue

for f in dog_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue

###################################
# Prepare train & test samples
###################################

# test-train split
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.5, random_state=42)

# encode the Ys
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)

###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cat_dog_6.tflearn', max_checkpoints=3,
                    tensorboard_verbose=3, tensorboard_dir='tmp/tflearn_logs/')

###################################
# Train model for 100 epochs
###################################
#model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,n_epoch=5, run_id='model_cat_dog_6_final', show_metric=True)

#model.save('model_cat_dog_6_final.tflearn')

model.load('model_cat_dog_6_final.tflearn')

def getImageinFile(dosyaadi):
    resim=cv2.imread("%s"%dosyaadi)
    return resim


def getOneHotLabel(imageName):
    word_label = imageName.split('/')[0]
    return word_label





def findDog(path):

    girisverisi = np.array([])
    klasordengelen = 0
    klasordengelen = getImageinFile(path)
    boyutlandirilmisresim = cv2.resize(klasordengelen, (64, 64))
    girisverisi = np.append(girisverisi, boyutlandirilmisresim)
    girisverisi = np.reshape(girisverisi, (-1, 64, 64, 3))
    doub = model.predict(girisverisi)
    print('%.8f' % (float(doub[0][1])))
    dogRate = float(doub[0][1])
    return dogRate

def findCat(path):

    girisverisi = np.array([])
    klasordengelen = 0
    klasordengelen = getImageinFile(path)
    boyutlandirilmisresim = cv2.resize(klasordengelen, (64, 64))
    girisverisi = np.append(girisverisi, boyutlandirilmisresim)
    girisverisi = np.reshape(girisverisi, (-1, 64, 64, 3))
    doub = model.predict(girisverisi)
    print('%.8f' % (float(doub[0][0])))
    catRate = float(doub[0][0])
    return catRate


def testFunc():
    files_path = 'D://test3/'
    cat_files_path = os.path.join(files_path, 'cat*.jpg')
    dog_files_path = os.path.join(files_path, 'dog*.jpg')
    cat_files = sorted(glob(cat_files_path))
    dog_files = sorted(glob(dog_files_path))
    value = 0.5
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n_files = len(cat_files) + len(dog_files)
    catCount = len(cat_files)+1
    dogCount = len(dog_files)+1
    for i in range(1, catCount):
        path = 'D://test3/cat.%s' % i
        uzanti = '.jpg'
        path = path + uzanti
        catRate = findCat(path)
        if catRate >= 0.5:
            tp = tp+1
        else:
            fp = fp+1

    print('------------------------------------------------')
    for i in range(1, dogCount):
        path = 'D://test3/dog.%s' % i
        uzanti = '.jpg'
        path = path + uzanti
        dogRate = findDog(path)
        if dogRate >= 0.5:
            tn = tn+1
        else:
            fn = fn+1
    confussionMatrix(tn, tp, fn, fp)

def confussionMatrix(tn, tp, fn, fp):
    totaltrue = tp + tn
    totalfalse = fn + fp
    total = totaltrue + totalfalse
    accuracy = (totaltrue / total) * 100
    err = (totalfalse / total) * 100
    tp1 = (tp / total) * 100
    tn1 = (tn / total) * 100
    fp1 = (fp / total) * 100
    fn1 = (fn / total) * 100
    print("Accuracy : %s " % accuracy)
    print("Error Rate : %s " % err)
    print("True Negative Rate : %s " % tn1)
    print("True Positive Rate : %s " % tp1)

    print("False Negative Rate : %s " % fn1)
    print("False Positive Rate : %s " % fp1)

    print("Confussion Matrix : ")

    print("                                            TAHMİN                 ")

    print("                                |     dog      |    cat   | ")
    print("GERCEK              | dog |           %s " % tn + "            " + "%s" % fp)

    print("                    | cat |           %s " % fn + "            " + "%s" % tp)


#tensorboard --logdir=C:/Users/veliyetisgengil/PycharmProjects/catsordogAug/tmp/tflearn_logs


testFunc()
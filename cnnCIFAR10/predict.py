import os

import matplotlib.pyplot as plt
import numpy as np
import tflearn
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import util

imgsize = (32, 32)

KAGGLE=False

if KAGGLE:
    modelid = "cnnKAGGLEDVC"
else:
    modelid = "cnnREDDITDVC"

imageset = util.datafolder + modelid + "\\TEST\\"
folder = util.modelfolder + modelid + "\\"
chkpath = folder + "checkpoint"
modelpath = folder + "model\\"
decoderfile = folder + "decoder.json"
n_categories = 10
savefigs=modelpath + "figs\\"

decoder = util.loaddecoder(decoderfile)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
indata = input_data(shape=[None, imgsize[0], imgsize[1], 3],
                    data_preprocessing=img_prep,
                    data_augmentation=img_aug)
conv1 = conv_2d(indata, 32, 3, activation='relu')
pool1 = max_pool_2d(conv1, 2)
conv2 = conv_2d(pool1, 64, 3, activation='relu')
conv3 = conv_2d(conv2, 64, 3, activation='relu')
pool2 = max_pool_2d(conv3, 2)
conn1 = fully_connected(pool2, 512, activation='relu')
drop1 = dropout(conn1, 0.5)
conn2 = fully_connected(drop1, n_categories, activation='softmax')
network = regression(conn2, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.005)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path=chkpath)

# if not os.path.exists(newsavepath):
#    shutil.copytree(savepath, newsavepath)
model.load(modelpath)

imagestream = util.Loader(imageset, onehot=False, size=imgsize, decoder=decoder)


if not os.path.exists(savefigs):
    os.mkdir(savefigs)

count = 0
imgs, labels, originals = imagestream.batchwithimg(size=10000)
preds = model.predict(imgs)

s = 0
for i in range(len(preds)):
    if util.maxprob(preds[i]) == labels[i]:
        s += 1

print("Results: " + str(s) + "/" + str(len(preds)) + ", " + str(100. * s / len(preds)) + "%.")

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

REDDITMODEL = False
REDDITDATA = False
if REDDITMODEL:
    modelid = "cnnCIFARDVC"
else:
    modelid = "cnnREDDITDVC"

if REDDITDATA:
    data = "cnnCIFARDVC"
else:
    data = "cnnREDDITDVC"

KAGGLE=False

if KAGGLE:
    modelid = "cnnKAGGLEDVC"
else:
    modelid = "cnnREDDITDVC"
imgsize = (64, 64)
imageset = util.datafolder + modelid + "\\TEST\\"
folder = util.modelfolder + modelid + "\\"

savefigs = folder + "figs/"
savepath = folder + "model/"
newsavepath = folder + "model/"  # "./tmp/model-pred"
chkpath = folder + "checkpoint"
logpath = folder + "log/"
modelid = "dvcCovnet"
modelpath = folder + modelid + ".tfl"
decoderfile = newsavepath + "decoder.json"
filename = "./dvccovnet/dvcCovnet.tfl"
n_categories = 2
testname = "G:\\RIR\\dogsncats\\dog\\t3_2s277m.png"

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
                     learning_rate=0.)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path=chkpath)

print(savepath)
# if not os.path.exists(newsavepath):
#    shutil.copytree(savepath, newsavepath)
model.load(newsavepath)
imagestream = util.Loader(imageset, onehot=False, size=imgsize, decoder=decoder)


if not os.path.exists(savefigs):
    os.mkdir(savefigs)





icount = 0
batchsize = 100
#totalimages = 25000
totalimages = 5000
preds = []
labels = []
while icount < totalimages:
    imgs, batchlabels, originals = imagestream.batchwithimg(size=batchsize)

    labels += np.reshape(batchlabels, (batchsize,)).tolist()
    predbatch=model.predict(imgs)
    preds += [util.maxprob(np.reshape(x, (2, ))) for x in predbatch]
    icount += batchsize
    print(str(icount) + "/" + str(totalimages))

correctdogs = 0
wrongdogs = 0
correctcat = 0
wrongcat = 0

for label, pred in zip(labels, preds):
    if decoder[label] == 'dog':
        if pred == label:
            correctdogs += 1
        else:
            wrongdogs += 1

    else:
        if pred == label:
            correctcat += 1
        else:
            wrongcat += 1


print("Correct dogs: " + str(correctdogs) + "/" + str(correctdogs + wrongdogs))
print("Correct cats: " + str(correctcat) + "/" + str(correctcat + wrongcat))





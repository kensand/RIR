import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os

import util

def main():

    modelid = "cnnKAGGLEDVC"
    imageset = util.datafolder + modelid + "\\TRAIN\\"
    folder = util.modelfolder + modelid + "\\"


    savepath = folder + "model/"
    chkpath = savepath + "chkpnt/"
    decoderfile = savepath + "decoder.json"
    img_size = (64, 64)

    learning_rate = 0.0005

    loader_batch_size = 2500
    train_batch_size = 100
    n_categories = 2
    train_steps = 20000
    epochs = 20

    if not os.path.exists(folder):
        os.mkdir(folder)
    decoder = None
    madedir = False
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        madedir = True
    else:
        decoder = util.loaddecoder(decoderfile)
    # imagestream = util.Loader("G:\\RIR\\dogsncats", onehot=True, size=(64, 64))
    # #(categoriesToSubs={'cat':['cats'], 'dog':['dogpictures']},
    #  size=img_size, onehot=True, imgbuffersize=1000, urlbuffsize=5000)
    # imagestream = util.RandLoader("G:\\RIR\\PetImages", onehot=True, size=img_size)

    imagestream = util.Loader(imageset, onehot=True, size=img_size, decoder=decoder)
    print(imagestream.getdecoder())
    util.storedecoder(decoder=imagestream.getdecoder(), filename=decoderfile)

    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Convolutional network building
    network = input_data(shape=[None, img_size[0], img_size[1], 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, n_categories, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)

    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path=chkpath, max_checkpoints=3)

    if os.path.exists(savepath) and not madedir:
        model.load(savepath)

    count = 0
    while count < train_steps:
        batch_x, batch_y = imagestream.batch(size=loader_batch_size)

        model.fit(batch_x, batch_y, n_epoch=epochs, shuffle=True, validation_set=0.,
                  show_metric=True, batch_size=train_batch_size, run_id=modelid)

        model.save(savepath)
        print("Model saved at: " + savepath)

        del batch_x
        del batch_y
        count += loader_batch_size
        if count % 100 == 0:
            print('Completed training on image number ', count)

    model.save(savepath)

if __name__ == "__main__":
    main()

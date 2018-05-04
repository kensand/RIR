import util
import os
import PIL.Image
import numpy as np


def convertcifar10(cifarpath, savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    batches = []
    for i in range(1, 6):
        batches.append(util.unpickle(cifarpath + "data_batch_" + str(i)))
    batches.append(util.unpickle(cifarpath + "test_batch"))

    meta = util.unpickle(cifarpath + "batches.meta")
    names = meta[b"label_names"]
    count = 0

    for batch in batches:
        data = batch[b"data"]
        labels = batch[b"labels"]

        for j in range(len(data)):
            p = savepath + names[labels[j]].decode('utf-8')
            if not os.path.exists(p):
                os.mkdir(p)

            ar = np.reshape(data[j], (3, 1024))
            ar = np.transpose(ar, (1, 0))
            ar = np.reshape(ar, (32, 32, 3))
            im = PIL.Image.fromarray(ar, "RGB")
            im.save(open(p + "\\" + str(count) + ".png", "wb"))
            count += 1
            print(str(count))


if __name__ == "__main__":
    convertcifar10("G:/RIR/cifar-10-batches-py/", "G:/RIR/CIFAR10/")

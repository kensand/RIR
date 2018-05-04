import io
import json
import os
import random
import time
import urllib.parse
import urllib.request

import PIL.Image
import numpy as np
import praw
import botCredentials

from config import *


redditStartTime = 1415463675
reddit = praw.Reddit(client_id=botCredentials.client_id, client_secret=botCredentials.client_secret,
                     user_agent="RedditImageRecognition 0.1")

random.seed(time.time())

supportedImages = [".jpg", ".png"]
supportedGifs = [".gif"]


def maxprob(pred):
    best = 0
    count = 0.
    for index, prob in enumerate(pred):
        if prob > count:
            count = prob
            best = index

    return best


def isimage(url):
    # print(url[-4:].lower(), supportedImages, (url[-4:].lower() in supportedImages))
    return url[-4:].lower() in supportedImages

imgurlocs = ['i.imgur.com', 'imgur.com']
def isimgur(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc not in imgurlocs:
        return False
    p = parsed.path
    l = os.path.split(p)
    if  len(l) < 2 :
        return False
    if l[0] == '/':
        1




def isgif(url):
    # print(url[-4:].lower(), supportedImages, (url[-4:].lower() in supportedImages))
    return url[-4:].lower() in supportedGifs

def getgifimgs():
    12


def getimage(imageurl):
    imfile = urllib.request.urlopen(imageurl)
    im = io.BytesIO(imfile.read())
    return PIL.Image.open(im)


def getimage(imageurl):
    imfile = urllib.request.urlopen(imageurl)
    im = io.BytesIO(imfile.read())
    return PIL.Image.open(im)

class UnlimitedSub:
    def __init__(self, subreddit, stop=False):
        self.stop = stop
        self.sub = reddit.subreddit(subreddit)
        self.currentt = int(time.time())
        self.currentgen = self.sub.submissions(end=self.currentt)
        self.lastt = self.currentt

    def next(self):
        ret = None
        reset = False
        while ret is None and self.currentt > redditStartTime:
            try:
                ret = self.currentgen.__next__()
            except StopIteration:
                ret = None
                if self.lastt == self.currentt:
                    if self.stop or reset:
                        raise StopIteration()
                    else:
                        reset = True
                        self.reset()

                self.currentt = self.lastt
                self.currentgen = self.sub.submissions(end=self.currentt)
            except Exception as e:
                ret = None
                raise StopIteration(e)

            if ret is not None:
                self.lastt = int(ret.created)
        return ret

    def __next__(self):
        return self.next()

    def batch(self, size=10):
        return [self.next() for _ in range(size)]

    def reset(self):
        self.currentt = int(time.time())
        self.currentgen = self.sub.submissions(end=self.currentt)
        self.lastt = self.currentt


def processgif(im):
    i = 0
    mypalette = im.getpalette()
    ims = []
    try:
        while 1:
            im.putpalette(mypalette)
            new_im = PIL.Image.new("RGBA", im.size)
            new_im.paste(im)
            ims.append(new_im)
            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass  # end of sequence

    return ims


def createdataset(drive, folder, categorysubs, categorysize=100, dims=None, includegifs=False, maxdups=1000):
    absp = os.path.join(drive, os.sep, folder)
    if not os.path.exists(absp):
        os.makedirs(absp)
    if os.listdir(absp):
        raise ValueError("Data Directory not empty: " + drive + folder)

    visited = []

    for category in categorysubs.keys():
        p = os.path.join(absp, category)
        os.makedirs(p)
        subs = [UnlimitedSub(sub) for sub in categorysubs[category]]
        count = 0
        dupcount = 0
        while count < categorysize:
            if len(subs) == 0:
                print("ran out of subreddits for the category: " + str(category) + " with " + str(count) + " images.")
                break
            s = random.choice(subs)

            try:
                post = s.next()
            except Exception as e:
                print(e)
                post = None
            if post is None:
                subs.remove(s)
            elif post.fullname not in visited:
                visited.append(post.fullname)
                if isimage(post.url):
                    try:
                        img = getimage(post.url)
                    except Exception as e:
                        img = None
                        print(e)
                    if img is not None:
                        try:
                            name = os.path.join(p, post.fullname + ".png")
                            if dims is not None:
                                img = img.resize(dims)
                            img = img.convert(mode="RGB")
                            img.save(open(name, "wb"), "PNG")
                            count += 1
                        except Exception as e:
                            print(e)

                elif includegifs and isgif(post.url):
                    try:
                        gif = getimage(post.url)
                    except Exception as e:
                        gif = None
                        print(e)
                    if gif is not None:
                        try:
                            imgs = processgif(gif)
                            c = 1
                            if len(imgs) > 5:
                                spacing = int(len(imgs) / 5)
                            else:
                                spacing = 1
                            i = 0
                            nims = []
                            while len(nims) < 5 and i < len(imgs):
                                nims.append(imgs[i])
                                i += spacing
                            imgs = nims
                            for img in imgs:
                                name = os.path.join(p, post.fullname + "_frame" + str(c) + ".png")
                                img = img.resize(dims)
                                img = img.convert(mode="RGB")
                                img.save(open(name, "wb"), "PNG")
                                c += 1
                                count += 1
                        except Exception as e:
                            print(e)
                print(str(category) + ": " + str(count) + "/" + str(categorysize))

            else:
                dupcount += 1
                if dupcount < maxdups:
                    print("Visited same post twice, for the " + str(dupcount) + " time. Will stop search for images in " + str(category) + " if " + str(maxdups) + " duplicates are found in total.")
                else:
                    print("Ran out of new images for " + str(category) + ",  category. Found " + str(count) + " images in total.")
                    dupcount = 0
                    break



class Loader:
    def __init__(self, folder, onehot=False, size=(64, 64), decoder=None, dtype=np.double):
        subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
        self.dtype = dtype
        self.catimgs = {}
        self.cattoint = {}
        self.intotcat = {}
        self.onehot = onehot
        self.size = size
        count = 0
        for cat in subfolders:
            self.catimgs[cat] = [os.path.join(folder, cat, file) for file in os.listdir(os.path.join(folder, cat))]
            if decoder is None:
                self.cattoint[cat] = count
                self.intotcat[count] = cat
            count += 1

        if decoder is not None:
            encoder = {v: k for k, v in decoder.items()}
            for key in self.catimgs.keys():
                if key not in encoder:
                    raise ValueError("Missing category " + str(key) + " in decoder: " + str(decoder))
            self.cattoint = encoder
            self.intotcat = decoder
        self.catimgscopy = {k: v.copy() for k, v in self.catimgs.items()}

    def reset(self):
        self.catimgscopy = {k: v.copy() for k, v in self.catimgs.items()}

    def getdecoder(self):
        return self.intotcat

    def nextwithimg(self):
        if len(self.catimgscopy) == 0:
            self.reset()
            if len(self.catimgscopy) == 0:
                raise RuntimeError("Loading from empty data set.")

        cat = random.choice(list(self.catimgscopy.keys()))
        file = random.choice(self.catimgscopy[cat])
        self.catimgscopy[cat].remove(file)
        if len(self.catimgscopy[cat]) == 0:
            self.catimgscopy.pop(cat)
        fp = open(file, "rb")
        im = PIL.Image.open(fp)
        im = im.convert(mode="RGB")
        im = im.resize(self.size)
        ret = np.array(im, dtype=self.dtype)

        if self.onehot:
            rlabel = [0] * len(self.catimgs)
            rlabel[self.cattoint[cat]] = 1
        else:
            rlabel = self.cattoint[cat]
        return ret, np.array(rlabel), im

    def batchwithimg(self, size=100):
        retdats = []
        retcats = []
        imgs = []
        count = 0
        while count < size:
            try:
                rd, rc, im = self.nextwithimg()
            except Exception as e:
                print(e)
                rd = None
                rc = None
                im = None
            if rd is not None and rc is not None and im is not None:
                count += 1
                retdats.append(rd)
                retcats.append(rc)
                imgs.append(im)
                del rd
                del rc
                del im
        return np.array(retdats), np.array(retcats), imgs

    def next(self):
        if len(self.catimgscopy) == 0:
            self.reset()
            if len(self.catimgscopy) == 0:
                raise RuntimeError("Loading from empty data set.")
        cat = random.choice(list(self.catimgscopy.keys()))
        while len(self.catimgscopy[cat]) == 0:
            self.catimgscopy.pop(cat)
            if len(self.catimgscopy) == 0:
                self.reset()
                if len(self.catimgscopy) == 0:
                    raise RuntimeError("Loading from empty data set.")
            cat = random.choice(list(self.catimgscopy.keys()))

        file = random.choice(self.catimgscopy[cat])
        self.catimgscopy[cat].remove(file)

        fp = open(file, "rb")
        im = PIL.Image.open(fp)
        im = im.convert(mode="RGB")
        im = im.resize(self.size)
        ret = np.array(im, dtype=self.dtype)

        if self.onehot:
            rlabel = [0] * len(self.catimgs)
            rlabel[self.cattoint[cat]] = 1
        else:
            rlabel = self.cattoint[cat]
        return ret, np.array(rlabel)

    def batch(self, size=100):
        retdats = []
        retcats = []
        count = 0
        while count < size:
            try:
                rd, rc = self.next()
            except Exception as e:
                print(e)
                rd = None
                rc = None
            if rd is not None and rc is not None:
                count += 1
                retdats.append(rd)
                retcats.append(rc)
                del rd
                del rc

        return np.array(retdats), np.array(retcats)


def storedecoder(decoder, filename):
    json.dump(decoder, open(filename, "w"))


def loaddecoder(filename):
    decoder = json.loads(open(filename, "r").read())
    print(decoder)
    return {int(key): value for key, value in dict(decoder).items()}


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

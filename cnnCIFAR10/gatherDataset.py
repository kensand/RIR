import util
import subredditcategories

if __name__ == "__main__":
    # createDataSet("G:", "\\RIR\\dogsncats",
    # {'cat': ['cats'], 'dog': ['dogpictures', 'puppysmiles']}, categorysize=25000)
    util.createdataset(util.datadrive, util.datafoldername + "cnnREDDIT10\\", subredditcategories.CATEGORY_SUBREDDITS_CIFAR10,
                       categorysize=6000, dims=(32,32))

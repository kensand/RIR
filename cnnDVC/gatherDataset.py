import util
import subredditcategories
if __name__ == "__main__":
    # createDataSet("G:", "\\RIR\\dogsncats",
    # {'cat': ['cats'], 'dog': ['dogpictures', 'puppysmiles']}, categorysize=25000)
    util.createdataset(util.datadrive, util.datafoldername + "cnnREDDITDVC\\", subredditcategories.CATEGORY_SUBREDDITS_DOGSVSCATS,
                       categorysize=25000, dims=(64, 64))

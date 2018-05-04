import util
import subredditcategories

if __name__ == "__main__":
    util.createdataset(util.datadrive, util.datafoldername + "inceptionREDDITDVC\\",
                       subredditcategories.CATEGORY_SUBREDDITS_DOGSVSCATS, categorysize=6000, dims=(277, 277))

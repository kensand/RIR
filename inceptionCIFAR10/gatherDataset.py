import util
import subredditcategories

if __name__ == "__main__":
    util.createdataset(util.datadrive, util.datafoldername + "inceptionREDDIT10\\", subredditcategories.CATEGORY_SUBREDDITS_CIFAR10,
                       categorysize=6000, dims=(300, 300))

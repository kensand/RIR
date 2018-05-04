# RIR
Reddit Image Recognizer (RIR) in python

This was an experiment for my Computer Vision class project.

Requires python 3.X, Tensorflow, tflearn, numpy, and PRAW.

NOTE: Unfortunately, the method of gathering datasets no longer works as of 4/11/18 because Reddit is mean and changed their API... Now gathering data in the same fashion is not possible (searching by date is no longer possible from the Reddit API). There are other methods available using other APIS to access Reddit, but I do not have time (nor a need) to do so as I had gathered all my data before this change.


In order to run:

0. Go to reddit.com, make an account, and see this page to make yourself a personal use script oauth token: https://www.reddit.com/prefs/apps . Then set client_id and client_secret in botCredentials.py so you can access Reddit's API.
1. See the config.py file in this folder in order to set your data and model locations folders.
2. Each subfolder contains a separate, runnable set of files that are capable of gathering the dataset, training a model on either the reddit dataset or the comparison dataset (CIFAR10 and Kaggle DVC, both are expected to be provided by the user), and predicting on the test set for evaluation.
3. Run the gatherDataset.py file in whatever folder you wish to test to generate the Reddit version of that dataset.
4. Manually split your image datasets into two sub folders: "TRAIN" and "TEST".
5. Run the training for the Reddit model and the comparison model. Make sure they are able to find your dataset and model folders. You may have to manually delete the created model folder if you wish to restart or there is an error.
6. Run the predict.py file to predict whichever model with whichever evaluation dataset. In order to change the model and dataset, edit the predict file.

This model can be extended easily by copying an existing model and customizing it to your liking. See subredditcategories.py for n idea on how to structure your input to the gatherDatatset function.

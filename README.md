# TrafficSignClassifier
Code for Traffic sign classifier

Please download dataset from here: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data#

# Files:
# loadDataFromFile.py 
Used to process through all the images in the above dataset and create an array of images against classes(Classification)
# classifierModel.py 
is the Convolutional Neural Network model built to recognize different traffic signs
# TrainModel.py 
used to loadData and train the CNN model and save to disk
# test_model.py 
used to test the generated model

# Sequence of running the pipeline:
Run TrainModel.py to train and save model to disk
<br>
Run test_model.py to test trained model against new Traffic Sign images

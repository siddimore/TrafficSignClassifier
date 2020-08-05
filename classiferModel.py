import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

class RoadSignClassifier:

  def CNN(width, height, depth, classes):
    model = Sequential()

    # Using 60 filters with stride of 5 and relu activation
    model.add(Conv2D(60,(5,5), input_shape= (32,32,3), activation = 'relu'))
    # add another convolution layer
    model.add(Conv2D(60,(5,5), input_shape= (32,32,3), activation = 'relu'))
    #pooling layer
    model.add(MaxPooling2D(pool_size = (2,2)))
    # add another convolutional layer
    model.add(Conv2D(30, (3, 3) , activation = 'relu'))
    model.add(Conv2D(30, (3, 3) , activation = 'relu'))
    # pooling layer
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    #Flatten the image to 1 dimensional array
    model.add(Flatten())
    #add a dense layer : amount of nodes, activation
    model.add(Dense(500, activation = 'relu'))
    # place a dropout layer
    #0.5 drop out rate is recommended, half input nodes will be dropped at each update
    model.add(Dropout(0.5))
    # defining the ouput layer of our network
    model.add(Dense(43, activation = 'softmax'))

    return model
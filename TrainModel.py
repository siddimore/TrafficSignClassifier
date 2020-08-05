from loadDataFromFile import load_data
from classiferModel import RoadSignClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

epochs = 20
learning_rate = 0.001
batch_size = 64

dataPath = '/Users/siddharthmore/trafficsignClassifier/trafficsign_images/'
train_data = dataPath + 'Train.csv'
test_data = dataPath + "Test.csv"
(trainX, trainY) = load_data(train_data)
(testX, testY) = load_data(test_data)

print("UPDATE: Normalizing data")
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

print("UPDATE: One-Hot Encoding data")
num_labels = len(np.unique(trainY))
print(num_labels)
trainY = to_categorical(trainY, num_labels)
testY = to_categorical(testY, num_labels)

class_totals = trainY.sum(axis=0)
class_weight = class_totals.max() / class_totals

print(class_totals)
print(class_weight)

data_aug = ImageDataGenerator(
rotation_range=10,
zoom_range=0.15,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.15,
horizontal_flip=False,
vertical_flip=False)

model = RoadSignClassifier.CNN(width=32, height=32, depth=3, classes=43)
optimizer = Adam(lr=learning_rate, decay=learning_rate / (epochs))
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
fit = model.fit_generator(
    data_aug.flow(trainX, trainY, batch_size=batch_size), 
    epochs=epochs,
    validation_data=(testX, testY),
    class_weight=class_weight,
    verbose=1)

score = model.evaluate(testX, testY , verbose = 0)
print("The test score is: ",score[0])
print("The Accuracy score is: ",score[1])

from keras.models import load_model

model.save('my_model.h5')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training', 'validation'])
# plt.title('Model Loss')
# plt.xlabel('epoch')
# model.save("model.h5")
# model.fit(
#     data_aug.flow(trainX, trainY, batch_size=batch_size), 
#     epochs=epochs,
#     validation_data=(testX, testY),
#     class_weight=class_weight,
#     verbose=1,
#     callbacks=[LearningRateScheduler(lr_schedule),
#                     ModelCheckpoint('model.h5',save_best_only=True)])


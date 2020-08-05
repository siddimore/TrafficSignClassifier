import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

dataPath = 'yourPath/trafficsign_images/'

col_names = ["Width","Height","Roi.X1","Roi.Y1","Roi.X2","Roi.Y2","ClassId","Path"]
def load_data(dataSet):
	images = []
	classes = []

	rows = pd.read_csv(dataSet, engine='python', names=col_names, error_bad_lines=False)
	rows = rows.sample(frac=1).reset_index(drop=True)

	for i, row in rows.iterrows():
		img_class = row["ClassId"]
		img_path = row["Path"]
		image = os.path.join(dataPath, img_path)
		ext = os.path.splitext(image)[-1].lower()
		if ext == ".png":
			image = cv2.imread(image)
			image_rs = cv2.resize(image, (32, 32), 3)
			R,G,B = cv2.split(image_rs)
			img_r = cv2.equalizeHist(R)
			img_g = cv2.equalizeHist(G)
			img_b = cv2.equalizeHist(B)
			new_image = cv2.merge((img_r, img_g, img_b))

			images.append(new_image)
			classes.append(img_class)

	X = np.array(images)
	y = np.array(classes)
    
	return (X, y)


import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import load_model



url = 'https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

img = np.asarray(img)
img_rs = cv2.resize(img, (32, 32), 3)

# image_rs = cv2.resize(img, (32, 32), 3)

#image_rs = cv2.resize(img, (32, 32), 3)

R,G,B = cv2.split(img_rs)
img_r = cv2.equalizeHist(R)
img_g = cv2.equalizeHist(G)
img_b = cv2.equalizeHist(B)
new_image = cv2.merge((img_r, img_g, img_b))
new_image = new_image.astype("float32")/255.0
new_image = new_image.reshape(-1,32,32,3)

#print(new_image.shape)
model = load_model('my_model.h5')
print("The Predicted sign: "+ str(model.predict_classes(new_image)))
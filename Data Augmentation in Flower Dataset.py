import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')

list(data_dir.glob('*/*.jpg'))[:5]

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))
roses[:5]

PIL.Image.open(str(roses[1]))

tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))

flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}

flowers_images_dict['roses'][:5]

str(flowers_images_dict['roses'][0])

img = cv2.imread(str(flowers_images_dict['roses'][0]))

img.shape

cv2.resize(img,(180,180)).shape
X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

num_classes = 5

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history=model.fit(X_train_scaled, y_train,validation_data=(X_test_scaled,y_test), epochs=30)

history2=model.evaluate(X_test_scaled,y_test)
print(history2)

plt.figure(figsize=(10,8))
plt.plot(np.arange(0,30),history.history['accuracy'],label='Training_accuracy')
plt.plot(np.arange(0,30),history.history['loss'],label='Training_loss')
plt.plot(np.arange(0,30),history.history['val_accuracy'],label='Validation_accuracy')
plt.plot(np.arange(0,30),history.history['val_loss'],label='validation_loss')
plt.xlabel('#Epochs')
plt.ylabel('Percentage')
plt.legend()
plt.title('Training accuracy and Loss plot')
plt.show()

predictions = model.predict(X_test_scaled)
predictions

score = tf.nn.softmax(predictions[0])

y_test[0]

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomZoom(0.1),
   layers.experimental.preprocessing.RandomRotation(0.1),
  ]
)

plt.axis('off')
print("Image Dimensions were: {0}x{1}".format(X.shape[0], X.shape[1]))
plt.imshow(X[0])

plt.axis('off')
for i in range(9):
    img=data_augmentation(X)[0].numpy().astype("uint8")
    plt.imshow(img)
print("Resized Dimensions were: {0}x{1}".format(img.shape[0], img.shape[1]))

num_classes = 5

model = Sequential([
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train,validation_data=(X_test_scaled,y_test),epochs=30)

plt.figure(figsize=(10,8))
plt.plot(np.arange(0,30),history.history['accuracy'],label='Training_accuracy')
plt.plot(np.arange(0,30),history.history['loss'],label='Training_loss')
plt.plot(np.arange(0,30),history.history['val_accuracy'],label='Validation_accuracy')
plt.plot(np.arange(0,30),history.history['val_loss'],label='validation_loss')
plt.xlabel('#Epochs')
plt.ylabel('Percentage')
plt.legend()
plt.title('Training accuracy and Loss plot')
plt.show()

model.evaluate(X_test_scaled,y_test)

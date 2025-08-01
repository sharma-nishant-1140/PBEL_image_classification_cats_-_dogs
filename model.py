import kagglehub
path = kagglehub.dataset_download("salader/dogs-vs-cats")
print("Path to dataset files:", path)

import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os

train_dir = os.path.join(path, 'train')
test_dir = os.path.join(path, 'test')

IMG_SIZE = 128
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(train_dir, target_size = (IMG_SIZE, IMG_SIZE), batch_size = BATCH_SIZE, class_mode = 'binary', subset = 'training')

val_generator = train_datagen.flow_from_directory(train_dir, target_size = (IMG_SIZE, IMG_SIZE), batch_size = BATCH_SIZE, class_mode = 'binary', subset = 'validation')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(test_dir, target_size = (IMG_SIZE, IMG_SIZE), batch_size = BATCH_SIZE, class_mode = 'binary')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_generator, validation_data = val_generator, epochs = 10)

import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

model.save("dog_cat_classifier.h5")

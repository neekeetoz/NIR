import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import itertools

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, \
    accuracy_score, label_ranking_average_precision_score, classification_report, roc_curve, auc

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping, ModelCheckpoint


data_dir = 'D:/Users/inet/Documents/GitHub/Nir_all/Nikita_K/GTSRB_Challenge/train_for_cnn'
categories = ['priority_signs', 'warning_signs', 'mandatory_signs']
img_size = 32
for i in categories:
    path = os.path.join(data_dir, i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        image_array = cv2.resize(img_array, (img_size, img_size))

train_data = []

for i in categories:
    train_path = os.path.join(data_dir, i)
    tag = categories.index(i)
    for img in os.listdir(train_path):
        image_arr = cv2.imread(os.path.join(train_path, img), cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('1', image_arr)
        # cv2.waitKey(0)
        new_image_array = cv2.resize(image_arr, (img_size, img_size))
        train_data.append([new_image_array, tag])

X = []
y = []

for i, j in train_data:
    X.append(i)
    y.append(j)
X = np.array(X).reshape(-1, img_size, img_size)
print(X.shape)
X = X / 255.0
X = X.reshape(-1, 32, 32, 1)

y_enc = to_categorical(y, num_classes=3)

print(y_enc)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding = 'Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='Same', activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(3, activation="softmax"))

model.summary()

model.compile(optimizer=Adam(lr=1e-5), loss="categorical_crossentropy", metrics=['accuracy'])

epochs = 1

es = EarlyStopping(monitor='val_acc', mode='max', patience=3)

batch_size = 16
imggen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        zoom_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=True,
        vertical_flip=False)


imggen.fit(X_train)
history = model.fit_generator(imggen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_val, y_val),
                              steps_per_epoch=X_train.shape[0] // batch_size)

score = model.evaluate(X_test, y_test, verbose=0)
print('Metrix =  ', score[1])
predictions = model.predict(X_test, batch_size=32)
print('LRAP =  ', label_ranking_average_precision_score(y_test, predictions))

plt.plot(history.history["loss"], c="red")
plt.plot(history.history["val_loss"], c="black")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["train", "test"])
plt.rcParams["figure.figsize"] = (10, 10)
plt.show()

plt.plot(history.history["acc"], c="blue")
plt.plot(history.history["val_acc"], c="green")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["train", "test"])
plt.rcParams["figure.figsize"] = (50, 50)
plt.show()

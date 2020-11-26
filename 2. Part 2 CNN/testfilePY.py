
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator



train_images = []       
train_labels = []
shape = (224,224)  
train_path = 'train path goes here!'
for filename in os.listdir('train path goes here!'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(train_path,filename))
        
        train_labels.append(filename.split('_')[0])
        img = cv2.resize(img,shape)
        train_images.append(img)
train_labels = pd.get_dummies(train_labels).values
train_images = np.array(train_images)
test_images = []
test_labels = []
shape = (224,224)
test_path = 'test path goes here!'
for filename in os.listdir('test path goes here!'):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(test_path,filename))
        test_labels.append(filename.split('_')[0])
        img = cv2.resize(img,shape)
        test_images.append(img)        
test_images = np.array(test_images)
test_labels = pd.get_dummies(test_labels).values

_x_train = train_images
x_test = test_images
_y_train = train_labels
y_test = test_labels

x_train, x_val, y_train, y_val = train_test_split(_x_train, _y_train, test_size=0.1, random_state=1)
print("x train shape :",x_train.shape)
print("x val shape :",x_val.shape)
print("y train shape :",y_train.shape)
print("y val shape :",y_val.shape)



datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


model_s = tf.keras.Sequential()
model_s.add(tf.keras.layers.Conv2D(32, (3, 3), 
    activation='relu', input_shape=(224, 224, 3)))
model_s.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model_s.add(tf.keras.layers.Conv2D(32, (3, 3), 
    activation='relu'))
model_s.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model_s.add(tf.keras.layers.Conv2D(64, (3, 3), 
    activation='relu'))
model_s.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model_s.add(tf.keras.layers.Flatten())
model_s.add(tf.keras.layers.Dense(64, activation='relu'))
model_s.add(tf.keras.layers.Dropout(0.5))
model_s.add(tf.keras.layers.Dense(4, activation='softmax'))
model_s.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])
model_s.summary()


x_train=x_train/255.0
x_val=x_val/255.0


hist=model_s.fit(x_train,y_train,batch_size=100,epochs=5,validation_data=(x_val,y_val))

print(hist.history.keys())


fig, ax = plt.subplots()
ax.plot(hist.history["loss"],color="green",label="Training Loss")
ax.plot(hist.history["val_loss"],color="red",label="Validation Loss")
ax.set_title("Loss Plot")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss Values")
fig.show()


fig2, ax2 = plt.subplots()
ax2.plot(hist.history["accuracy"],color="black",label="Training Accuracy")
ax2.plot(hist.history["val_accuracy"],color="blue",label="Validation Accuracy")
ax2.set_title("Accuracy Plot")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy Values")
fig2.show()


prediction=model_s.predict(x_val)


from copy import deepcopy
predicted_classes=deepcopy(prediction)
predicted_classes=np.argmax(predicted_classes,axis=1)
y_true=np.argmax(y_val,axis=1)
print("y predicted classes shape :",predicted_classes.shape)
print("y true shape :",y_true.shape)


score = model_s.evaluate(test_images, test_labels)
print(score)
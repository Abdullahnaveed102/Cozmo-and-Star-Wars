import struct
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

#======================================================================================================>
#===================================================================================> Image folder path
data_dir = 'Pictures'

#======================================================================================================>
#==================================================================================> ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

#======================================================================================================>
#====================================================================> Training and validation datasets
# Training generator
batch_size = 32
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(320, 240),  # Adjust the size as needed
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(320, 240),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

class_indices_train = train_generator.class_indices
print("Class Indices (Training):", class_indices_train)

# Validation generator
class_indices_val = validation_generator.class_indices
print("Class Indices (Validation):", class_indices_val)

#======================================================================================================>
#====================================================================================> Simple CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(320, 240, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Assuming you have 5 characters

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#======================================================================================================>
#============================================================================================> Training
epochs = 10
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

#======================================================================================================>
#============================================================================================> Evaluate
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Test accuracy: {test_acc}')

#======================================================================================================>
#==========================================================================================> Save model
model.save('cozmo_model.h5')
print("Model saved successfully!")

#======================================================================================================>
#======================================================================================================>
#======================================================================================================>
#=============================================================================================> Testing
import numpy as np
from keras.preprocessing import image

#====================================> test1
img_path = 'test1.png'
img = image.load_img(img_path, target_size=(320, 240))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f'Predicted class: {predicted_class}')

#====================================> test2
img_path = 'test2.png'
img = image.load_img(img_path, target_size=(320, 240))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f'Predicted class: {predicted_class}')

#====================================> test3
img_path = 'test3.png'
img = image.load_img(img_path, target_size=(320, 240))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f'Predicted class: {predicted_class}')

#====================================> test4
img_path = 'test4.png'
img = image.load_img(img_path, target_size=(320, 240))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f'Predicted class: {predicted_class}')

#====================================> test5
img_path = 'test5.png'
img = image.load_img(img_path, target_size=(320, 240))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f'Predicted class: {predicted_class}')

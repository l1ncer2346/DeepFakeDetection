import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import codecs, json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
'''
obj_text = codecs.open('rgb.json', 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
a_new = np.array(b_new)
print(a_new)
'''
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
# стандартизация входных данных
x_train = x_train / 255
#x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 2)
#y_test_cat = keras.utils.to_categorical(y_test, 10)

#x_train = np.expand_dims(x_train, axis=3)
#x_test = np.expand_dims(x_test, axis=3)

print(x_train.shape)

model = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2,  activation='softmax')
])

# print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

#model.evaluate(x_test, y_test_cat)

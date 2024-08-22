import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train,y_train,epochs=3)

# model.save('digits.model')

# loss,accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy) -----> 0.97 %

model = tf.keras.models.load_model('digits.model')

image_number = 1

while os.path.isfile(f"digits/digit{image_number}.png"):
    img = cv.imread(f"digits/digit{image_number}.png", cv.IMREAD_GRAYSCALE)  # Read the image in grayscale
    img = cv.resize(img, (28, 28))  # Resize the image to (28, 28)
    img = np.invert(np.array([img]))  # Invert the image if necessary (depends on your dataset)
    
    prediction = model.predict(img)
    print(f"This digit is probably a {np.argmax(prediction)}")
    
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

    image_number += 1







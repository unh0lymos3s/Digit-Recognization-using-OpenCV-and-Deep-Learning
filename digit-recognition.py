import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

dataset = tf.keras.datasets.mnist
(xtrain, ytrain), (xtest, ytest) = dataset.load_data()

x_train = tf.keras.utils.normalize(xtrain, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
model.add(tf.keras.layers.Dense(256, activation = 'relu' ))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,ytrain,epochs=5)
model.save('digits.model')

samosa = tf.keras.models.load_model('digits.model')

loss, accuracy = samosa.evaluate(xtest,ytest)

print (loss)
print (accuracy)


image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = samosa.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1
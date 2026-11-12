# program 7

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and normalize MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32').reshape(-1, 28, 28, 1) / 255
x_test = x_test.astype('float32').reshape(-1, 28, 28, 1) / 255

# Build autoencoder
input_img = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, 3, activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(8, 3, activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D(2, padding='same')(x)

x = layers.Conv2D(8, 3, activation='relu', padding='same')(encoded)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(16, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)
decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder.fit(x_train, x_train, epochs=10, batch_size=128,
                shuffle=True, validation_data=(x_test, x_test))

# Reconstruct and display 10 images
decoded_imgs = autoencoder.predict(x_test)
plt.figure(figsize=(20, 4))

for i in range(10):
    for j, img, title in [
        (1, x_test[i].reshape(28, 28), 'Original'),
        (11, decoded_imgs[i].reshape(28, 28), 'Reconstructed')
    ]:
        ax = plt.subplot(2, 10, i + j)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

plt.tight_layout()
plt.show()

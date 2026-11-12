# program 3
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Flatten, Dense
from keras.utils import to_categorical

# Load data
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

# Reshape and normalize
train_X = train_X.reshape(-1, 28, 28, 1) / 255.0
test_X = test_X.reshape(-1, 28, 28, 1) / 255.0

# One-hot encoding
train_Y = to_categorical(train_Y, 10)
test_Y = to_categorical(test_Y, 10)

# CNN model
model = Sequential()

model.add(Conv2D(256, (3,3), padding='same', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(28, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_X, train_Y, epochs=5, batch_size=64)

# Test
loss, acc = model.evaluate(test_X, test_Y)
print("Test loss:", loss)
print("Test accuracy:", acc)

# Prediction
predictions = model.predict(test_X)
print(np.argmax(predictions[0]))

# Display image
plt.imshow(test_X[0].reshape(28,28), cmap='gray')
plt.show()
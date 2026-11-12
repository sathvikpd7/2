# program 4.1
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Load data
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255
test_X = test_X.reshape(-1, 28, 28, 1).astype('float32') / 255

train_Y = to_categorical(train_Y, 10)
test_Y = to_categorical(test_Y, 10)

# CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same',
           input_shape=(28,28,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    Conv2D(128, (3,3), activation='relu', padding='same'),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_X, train_Y,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

loss, acc = model.evaluate(test_X, test_Y)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

prediction = model.predict(test_X)
plt.imshow(test_X[0].reshape(28,28), cmap='gray')
plt.title(f"Predicted: {np.argmax(prediction[0])}")
plt.axis('off')
plt.show()
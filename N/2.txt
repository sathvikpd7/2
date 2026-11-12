# program 2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report

# Load MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Print shapes
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Display first 10 images
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(y_train[i])
    plt.axis('off')
plt.show()

# Reshape and normalize
X_train = X_train.reshape(60000, 784).astype("float32") / 255
X_test = X_test.reshape(10000, 784).astype("float32") / 255

print("New shape of X_train:", X_train.shape)
print("New shape of X_test:", X_test.shape)

# Create Deep Feed Forward Neural Network
model = Sequential([
    Dense(128, activation='relu', kernel_initializer='HeNormal', input_shape=(784,)),
    Dense(64, activation='relu', kernel_initializer='HeNormal'),
    Dense(32, activation='relu', kernel_initializer='HeNormal'),
    Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, batch_size=10, epochs=5)

# Predictions
pred_train = np.argmax(model.predict(X_train), axis=1)
pred_test = np.argmax(model.predict(X_test), axis=1)

# Model summary
print("\nModel Summary")
model.summary()

# Evaluation
print("\n--------- Evaluation on Training Data ---------")
print(classification_report(y_train, pred_train))

print("\n--------- Evaluation on Test Data ---------")
print(classification_report(y_test, pred_test))
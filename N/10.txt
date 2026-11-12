#program 10
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load data
vocab_size, maxlen, embedding_dim = 10000, 500, 32
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train and evaluate
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Prepare word index
word_index = {k: v + 3 for k, v in imdb.get_word_index().items()}
word_index.update({"<PAD>": 0, "<START>": 1, "<UNK>": 2, "<UNUSED>": 3})

# Encode and predict a sample review
sample_review = "This movie was fantastic! The story was compelling and the acting was great."
encoded = [1] + [word_index.get(w, 2) for w in sample_review.lower().split()]
processed = pad_sequences([encoded], maxlen=maxlen)
prediction = model.predict(processed, verbose=0)[0][0]

sentiment = "Positive" if prediction > 0.5 else "Negative"
print(f"\nSample review: {sample_review}")
print(f"Predicted Sentiment: {sentiment} ({prediction:.4f})")

from tensorflow.keras.datasets import imdb

(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = 10000)

import numpy as np

def vectorize_sequences(sequences, dimensions = 10000):
  results = np.zeros((len(sequences), dimensions))
  for i,sequences in enumerate(sequences):
    results[i, sequences] = 1
  return results

x_train = vectorize_sequences(train_data)
y_train = vectorize_sequences(test_data)


y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_label).astype('float32')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_shape=(10000, ), activation = "relu"))
model.add(Dense(16, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer='adam', loss = 'mse', metrics = ['accuracy'])

model.summary()

history = model.fit(x_train, y_train, validation_split = 0.3, epochs = 20, verbose = 1, batch_size = 512)


# Assuming you have already trained and compiled the model

# Vectorize the input sequence
def vectorize_sequence(sequence, dimensions=10000):
    vectorized = np.zeros((1, dimensions))
    for word_index in sequence:
        if word_index < dimensions:
            vectorized[0, word_index] = 1
    return vectorized

# Example review
review = [3, 25, 10, 156, 99]  # Replace with your own review or use test data

# Vectorize the review
vectorized_review = vectorize_sequence(review)

# Predict the sentiment
prediction = model.predict(vectorized_review)

# Determine if the review is positive or negative
sentiment = "Positive" if prediction[0] > 0.5 else "Negative"

print("Review sentiment:", sentiment)


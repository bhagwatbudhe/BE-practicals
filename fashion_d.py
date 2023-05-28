from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
import numpy as np
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# Build the model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_x.astype(np.float32), train_y.astype(np.float32), epochs=5, validation_split=0.2)

# Define class labels
labels = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

# Make predictions on the test data
predictions = model.predict(test_x)
predicted_labels = np.argmax(predictions, axis=1)
predicted_classes = [labels[label] for label in predicted_labels]

# Display some example images with their predicted categories
num_examples = 10
plt.figure(figsize=(12, 8))
for i in range(num_examples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_x[i], cmap='gray')
    plt.title(predicted_classes[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

import tensorflow as tf
from tensorflow import keras #type: ignore
from tensorflow.keras import layers #type: ignore

# Load dataset (MNIST: 0–9 handwritten digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # Input layer
    layers.Dense(128, activation='relu'),   # Hidden layer
    layers.Dense(64, activation='relu'),    # Hidden layer
    layers.Dense(10, activation='softmax')  # Output layer
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# Predict example
predictions = model.predict(x_test[:5])
print("Predicted labels:", predictions.argmax(axis=1))
print("Actual labels:", y_test[:5])
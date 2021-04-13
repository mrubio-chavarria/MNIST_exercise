
from keras.datasets import mnist
import tensorflow as tf
import numpy as np


if __name__ == '__main__':

    # Load data
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Convert the data to float point numbers
    train_x, test_x  = train_x / 255.0, test_x / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer='SGD',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


    # Train the model
    model.fit(train_x, train_y, epochs=5)

    # Test the model
    model.evaluate(test_x, test_y, verbose=2)

    # To make predictions
    # predictions = model(test_x)
    # print(predictions[0])
    # print(np.argmax(predictions[0]))
    # print(test_y[0])

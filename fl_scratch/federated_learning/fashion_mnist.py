import numpy as np
import tensorflow as tf

SEED = 2022


def load_model(input_shape=(28, 28, 1)):
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=SEED)

    inputs = tf.keras.layers.Input(shape=input_shape)
    layers = tf.keras.layers.Conv2D(
        32, kernel_size=(5, 5), strides=(1, 1),
        kernel_initializer=kernel_initializer, padding="same", activation="relu"
    )(inputs)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Conv2D(
        64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(layers)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Flatten()(layers)
    layers = tf.keras.layers.Dense(
        512, kernel_initializer=kernel_initializer, activation="relu"
    )(layers)

    outputs = tf.keras.layers.Dense(
        10, kernel_initializer=kernel_initializer, activation="softmax"
    )(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model


def load_data(partition: int, num_partitions: int):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Take a subset
    X_train, y_train = shuffle(X_train, y_train, seed=SEED)
    X_test, y_test = shuffle(X_test, y_test, seed=SEED)

    X_train, y_train = get_partition(X_train, y_train, partition, num_partitions)
    X_test, y_test = shuffle(X_test, y_test, partition, num_partitions)

    # Adjust x sets shape for model
    X_train = adjust_x_shape(X_train)
    X_test = adjust_x_shape(X_test)

    # Normalize data
    X_train = x_train.astype("float32") / 255.0
    X_test = x_test.astype("float32") / 255.0

    # Convert class vectors to one-hot encoded labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (X_train, y_train), (X_test, y_test)


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return cast(np.ndarray, nda_adjusted)


def shuffle(x_orig: np.ndarray, y_orig: np.ndarray, seed: int):
    """Shuffle x and y in the same way."""
    np.random.seed(seed)
    idx = np.random.permutation(len(x_orig))
    return x_orig[idx], y_orig[idx]


def get_partition(x_orig: np.ndarray, y_orig: np.ndarray, partition: int, num_clients: int):
    """Return a single partition of an equally partitioned dataset."""
    step_size = len(x_orig) / num_clients
    start_index = int(step_size * partition)
    end_index = int(start_index + step_size)
    return x_orig[start_index:end_index], y_orig[start_index:end_index]

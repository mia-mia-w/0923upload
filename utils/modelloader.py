from sklearn.neighbors import KNeighborsClassifier
from tensorflow.python.keras import Sequential, layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import (
    Conv1D,
    Conv2D,
    MaxPooling1D,
    MaxPooling2D,
)


def get_model(
        model_name: str, n_neighbors: int = 5, input_shape: int = 0, num_class: int = 0
):
    """
    Return a model with given configurations
    :param num_class: cnn number of classes
    :param input_shape: cnn input shape
    :param n_neighbors: number of neighbors in KNN model, default 5
    :param model_name: model name in abbr.
    :return: a model object
    """
    if model_name == "knn":
        model = knn_model(n_neighbors)
    elif model_name == "mlp":
        model = mlp_model()
    elif model_name == "cnn":
        model = cnn_model(input_shape, num_class)
    elif model_name == "rnn_lstm":
        model = rnn_lstm(input_shape, num_class)
    elif model_name == "cnn_mnist":
        model = cnn_mnist(input_shape, num_class)
    else:
        raise KeyError("Model name not found")
    return model


def knn_model(n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return model


def mlp_model():
    model = Sequential()
    model.add(layers.Input(2))
    model.add(layers.Dense(2, activation="relu"))
    model.add(layers.Dense(3, activation="relu"))
    model.add(layers.Dense(4))
    model.compile(optimizer="adam", loss="mae")
    model.build()
    print(model.summary())
    return model


def cnn_model(input_shape, num_class):
    model = Sequential()
    model.add(Conv1D(16, 3, activation="relu", input_shape=(input_shape, 1)))
    model.add(Conv1D(4, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(num_class, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print("CNN model loaded")
    return model


def rnn_lstm(input_shape, num_class):
    model = keras.Sequential()
    model.add(LSTM(16, input_shape=(input_shape, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(4, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(num_class, activation="softmax"))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def cnn_mnist(input_shape, num_class):
    print("Convolutional Neural Network Structure: ")
    model = Sequential()
    # model.add(Input(input_shape=input_shape))
    model.add(Conv2D(4, 3, strides=2, padding="same", activation="relu", input_shape=(input_shape, 1), ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation="softmax"))

    print(model.summary())

    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    return model

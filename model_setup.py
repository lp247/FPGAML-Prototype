import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def build_config(config):
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = train_data.astype('float32') / 255.0
    test_images = test_data.astype('float32') / 255.0
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    config["train_data"] = train_data
    config["train_labels"] = train_labels
    config["test_data"] = test_data
    config["test_labels"] = test_labels

    return config

def build_model(config):
    model = None
    model_path = config["name"] + ".keras"

    if config["load_saved_model"] and os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = config["model_shape"]
        model.compile(**config["compilation"])

        model.fit(
            config["train_data"],
            config["train_labels"],
            validation_data=(config["test_data"], config["test_labels"]),
            **config["fit"]
        )
        model.save(model_path)

    return model

from tensorflow import keras
from tensorflow.keras import layers

from typing import Tuple

def create_cnn(only_digits:bool = False, input_dim: Tuple[int, int, int] = (28,28,1)):
    inputs = layers.Input(shape=input_dim)
    x = layers.Conv2D(32, (3, 3))(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10 if only_digits else 62, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
from tensorflow import keras
from tensorflow.keras import layers

import functools

def create_lstm_shakespeare_v2(vocab_size: int, sequence_length: int, mask_zero:bool = True, seed:int=1111):
    if vocab_size < 1:
        raise ValueError('vocab_size must be a positive integer')
    if sequence_length < 1:
        raise ValueError('sequence_length must be a positive integer')
    
    model = keras.Sequential()
    model.add(
        layers.Embedding(
            input_dim=vocab_size,
            input_length=sequence_length,
            output_dim=8,
            embeddings_initializer=keras.initializers.RandomUniform(seed=seed),
    ))
    lstm_layer_builder = functools.partial(
        layers.LSTM,
        units=256,
        recurrent_initializer=keras.initializers.Orthogonal(seed=seed),
        kernel_initializer=keras.initializers.HeNormal(seed=seed),
        return_sequences=True,
        activation='tanh',
        stateful=False)
    model.add(lstm_layer_builder())
    model.add(lstm_layer_builder())
    model.add(layers.Dense(
        vocab_size,
        kernel_initializer= keras.initializers.GlorotNormal(
        seed=seed)))  # Note: logits, no softmax.
    return model

def create_lstm_shakespeare(vocab_size, sequence_length, embedding_dim=8, lstm_units=256):
    if vocab_size < 1:
        raise ValueError('vocab_size must be a positive integer')
    if sequence_length < 1:
        raise ValueError('sequence_length must be a positive integer')
        
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dense(vocab_size)
    ])
    return model
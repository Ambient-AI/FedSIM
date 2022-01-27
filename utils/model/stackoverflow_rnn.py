from tensorflow import keras
from tensorflow.keras import layers

def create_lstm_stackoverflow(vocab_size:int, 
    embedding_size:int = 96, 
    num_lstm_layers:int = 1,
    lstm_size:int = 670):
    if vocab_size < 1:
        raise ValueError('vocab_size must be a positive integer')

    inputs = layers.Input(shape=(None,))
    input_embedding = layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_size, mask_zero=False
    )
    embedded = input_embedding(inputs)
    projected = embedded

    for _ in range(num_lstm_layers):
        layer = layers.LSTM(lstm_size, return_sequences=True)
        processed = layer(projected)
        projected = layers.Dense(embedding_size)(processed)
    logits = layers.Dense(vocab_size)(projected)
    return keras.Model(inputs=inputs, outputs=logits)


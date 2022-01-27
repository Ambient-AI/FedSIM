from typing import Tuple
import logging

import tensorflow_federated as tff
import tensorflow as tf
import utils.models as models
import utils.keras_metrics as keras_metrics

DEFAULT_SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017

CHAR_VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
)
VOCAB_LENGTH = len(CHAR_VOCAB) + 4

DEFAULT_SHUFFLE_BUFFER_SIZE = 50

def get_special_tokens() -> Tuple[int, int, int, int]:
    """Token dataset preprocessing code

    Returns:
        A tuple containing the four types of special characters (pad, oov, bos, eos)
    
    """

    vocab_size = len(CHAR_VOCAB)
    pad = 0
    oov = vocab_size + 1
    bos = vocab_size + 2
    eos = vocab_size + 3
    return pad, oov, bos, eos

def _build_tokenize_fn(split_length: int = DEFAULT_SEQUENCE_LENGTH+1):
    """Convert a Shakespeare example into character ids

    Args:
        split_length: An integer used to determine the padding length for a given snippet.
        Pads until the sequence length is divisible by split_length.

    Returns:
        tf.function

    """
    _, _, bos, eos = get_special_tokens()

    ids = tf.range(len(CHAR_VOCAB), dtype=tf.int64)
    lookup_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(CHAR_VOCAB, ids), num_oov_buckets=1)
    
    def to_tokens_and_pad(example: tf.Tensor) -> tf.Tensor:
        """Convert Shakespeare example to int64 tensor of token ids and pad"""
        chars = tf.strings.bytes_split(example['snippets'])
        tokens = lookup_table.lookup(keys=chars) + 1
        tokens = tf.concat([[bos], tokens, [eos]], 0)
        pad_length = (-tf.shape(tokens)[0]) % split_length
        return tf.concat([tokens, tf.zeros(pad_length, dtype=tf.int64)], 0)
    return to_tokens_and_pad

def _split_target(sequence_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Split a N+1 Sequence into an input sequence of N characters, and and output 'predicted' character"""
    input_text = tf.map_fn(lambda x: x[:-1], sequence_batch)
    output_text = tf.map_fn(lambda x: x[1:], sequence_batch)
    return (input_text, output_text)

def create_preprocess_fn(num_epochs:int, batch_size:int,
                        sequence_length: int = DEFAULT_SEQUENCE_LENGTH, 
                        num_parallel_calls: int = tf.data.experimental.AUTOTUNE):
    """Creates a preprocessing function for Shakespeare client datasets"""
    if sequence_length < 1:
        raise ValueError('sequence_length must be a positive integer')
    
    shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE

    def preprocess_fn(dataset):
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)

        # Convert snippets to int64 tokens and pad
        to_tokens = _build_tokenize_fn(split_length=sequence_length+1)
        dataset = dataset.map(to_tokens, num_parallel_calls=num_parallel_calls)
        # Separate into individual tokens
        dataset = dataset.unbatch()
        # Join into sequences of the desired length
        dataset = dataset.batch(sequence_length+1, drop_remainder=True)
        dataset = dataset.batch(batch_size)

        return dataset.map(_split_target, num_parallel_calls=num_parallel_calls)
    return preprocess_fn

def create_task(num_epochs:int, batch_size:int = 4, sequence_length:int = DEFAULT_SEQUENCE_LENGTH):
    """Creates a preprocessed shakespeare NCP task"""
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    preprocessing_fn = create_preprocess_fn(num_epochs, batch_size)

    logging.info("Loading dataset from cache")
    train, test = tff.simulation.datasets.shakespeare.load_data()

    train = train.preprocess(preprocess_fn=preprocessing_fn)
    test = test.preprocess(preprocess_fn=preprocessing_fn)

    pad_token, _, _,_ = get_special_tokens()

    def model_fn():
        model = models.create_lstm_shakespeare(VOCAB_LENGTH, sequence_length)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics=[keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token])]

        return model, loss, metrics

    return train, test, model_fn
from typing import Tuple, List, Callable
import attr
import logging

import tensorflow_federated as tff
import tensorflow as tf

from utils.model.models import create_lstm_stackoverflow
from utils.model.keras_metrics import MaskedCategoricalAccuracy

DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_SHUFFLE_BUFFER_SIZE = 1000
DEFAULT_TAG_VOCAB_SIZE = 500
DEFAULT_WORD_VOCAB_SIZE = 10000

@attr.s(eq=False, frozen=True)
class SpecialTokens(object):
  """Structure for Special tokens.
  Attributes:
    padding: int - Special token for padding.
    out_of_vocab: list - Special tokens for out of vocabulary tokens.
    beginning_of_sentence: int - Special token for beginning of sentence.
    end_of_sentence: int - Special token for end of sentence.
  """
  padding = attr.ib()
  out_of_vocab = attr.ib()
  beginning_of_sentence = attr.ib()
  end_of_sentence = attr.ib()

  def get_number_of_special_tokens(self):
    return 3 + len(self.out_of_vocab)

def split_input_target(chunk: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Generates input and output data
    
    The task of stackoverflow is to predict the next word

    Args:
        chunk: A Tensor of text data

    Returns:
        A tuple of input and output data
    """
    input_text = tf.map_fn(lambda x: x[:-1], chunk)
    output_text = tf.map_fn(lambda x: x[1:], chunk)
    return (input_text, output_text)


def build_to_ids_fn(
    vocab: List[str],
    max_sequence_length: int,
    num_out_of_vocab_buckets: int = 1) -> Callable[[tf.Tensor], tf.Tensor]:
    """Constructs function mapping examples to sequences of token indices."""
    special_tokens = get_special_tokens(len(vocab), num_out_of_vocab_buckets)
    bos = special_tokens.beginning_of_sentence
    eos = special_tokens.end_of_sentence

    table_values = tf.range(len(vocab), dtype=tf.int64)
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(vocab, table_values),
        num_oov_buckets=num_out_of_vocab_buckets)

    def to_ids(example):
        sentence = tf.reshape(example['tokens'], shape=[1])
        words = tf.strings.split(sentence, sep=' ').values
        truncated_words = words[:max_sequence_length]
        tokens = table.lookup(truncated_words) + 1
        tokens = tf.cond(
            tf.less(tf.size(tokens), max_sequence_length),
            lambda: tf.concat([tokens, [eos]], 0), lambda: tokens)

        return tf.concat([[bos], tokens], 0)

    return to_ids


def batch_and_split(dataset: tf.data.Dataset, sequence_length: int,
                    batch_size: int) -> tf.data.Dataset:
    return dataset.padded_batch(batch_size, padded_shapes=[sequence_length + 1]).map(split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_special_tokens(vocab_size: int,
                       num_out_of_vocab_buckets: int = 1) -> SpecialTokens:
    """Gets tokens dataset preprocessing code will add to Stackoverflow."""
    return SpecialTokens(
        padding=0,
        out_of_vocab=[
            vocab_size + 1 + n for n in range(num_out_of_vocab_buckets)
        ],
        beginning_of_sentence=vocab_size + num_out_of_vocab_buckets + 1,
        end_of_sentence=vocab_size + num_out_of_vocab_buckets + 2)

def create_preprocess_fn(batch_size:int,
                        vocab: List[str], sequence_length:int = DEFAULT_SEQUENCE_LENGTH,
                        num_out_of_vocab_buckets:int = 1, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """Creates a preprocessing function for StackOverflow next word prediction"""
    if not vocab:
        raise ValueError("Vocab must be non-empty")
    if sequence_length < 1:
        raise ValueError('sequence_length must be a positive integer')
    if num_out_of_vocab_buckets <= 0:
        raise ValueError("num_out_of_vocab_buckets must be a positive integer")
    
    shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE
    to_ids = build_to_ids_fn(vocab=vocab, max_sequence_length=sequence_length, num_out_of_vocab_buckets=num_out_of_vocab_buckets)

    def preprocess_fn(dataset):
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(to_ids, num_parallel_calls=num_parallel_calls)
        return batch_and_split(dataset, sequence_length, batch_size)

    return preprocess_fn


def create_task(batch_size:int = 16, 
                sequence_length:int = DEFAULT_SEQUENCE_LENGTH,
                vocab_size:int = DEFAULT_WORD_VOCAB_SIZE,
                num_out_of_vocab_buckets:int = 1):
    """Creates a preprocessed stackoverflow NWP task"""
    logging.info("Loading vocab dictionary from Google Storage API")
    vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
    vocab = list(vocab_dict.keys())[:vocab_size]

    logging.info("Initializing preprocess functions")
    preprocessing_fn = create_preprocess_fn(batch_size, vocab, sequence_length, num_out_of_vocab_buckets=num_out_of_vocab_buckets)
    special_tokens = get_special_tokens(vocab_size, num_out_of_vocab_buckets)
    pad_token = special_tokens.padding
    oov_tokens = special_tokens.out_of_vocab
    eos_token = special_tokens.end_of_sentence

    def unbatch(dataset):
        return dataset.unbatch()

    def metrics_builder():
        return [MaskedCategoricalAccuracy(name='accuracy', masked_tokens=[pad_token]),
                MaskedCategoricalAccuracy(name='acc w/o oov', masked_tokens=[pad_token]+oov_tokens),
                MaskedCategoricalAccuracy(name='acc w/o oov or eos', masked_tokens=[pad_token, eos_token] + oov_tokens)]

    extended_vocab_size = (vocab_size + special_tokens.get_number_of_special_tokens())

    logging.info("Loading dataset from cache")
    train, _, test = tff.simulation.datasets.stackoverflow.load_data()

    logging.info("Preprocessing Stackoverflow data")
    train = train.preprocess(preprocess_fn=preprocessing_fn)
    test = test.preprocess(preprocess_fn=preprocessing_fn).preprocess(unbatch)

    element_spec = train.create_tf_dataset_for_client(train.client_ids[0]).element_spec

    def model_fn():
        model = create_lstm_stackoverflow(extended_vocab_size)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #metrics = metrics_builder()
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]

        return tff.learning.from_keras_model(
            keras_model=model,
            loss=loss,
            input_spec= element_spec,
            metrics=metrics
        )

    return train, test, model_fn
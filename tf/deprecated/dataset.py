import random
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import pickle
import logging


class Dataset:
    def __init__(self, server_prop=0, inner_rounds=1, train_batch_size=64, test_batch_size=128):
        self.server_prop = server_prop
        self.inner_rounds = inner_rounds
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.list_clients = None

    def get_client_data(self, client_id):
        data = self.train_data.create_tf_dataset_for_client(client_id)
        data = self.prefetch_batch(data, 16)
        return data

    def get_server_data(self, server_id):
        data = self.train_data.create_tf_dataset_for_client(server_id)
        data = self.prefetch_batch(data, 16)
        return data

    
    def get_test_data(self, client_id):
        data = self.test_data.create_tf_dataset_for_client(client_id)
        return self.prefetch_batch(data, self.test_batch_size)



    def get_server_proportion(self):
        return self.server_prop

class Shakespeare_Data(Dataset):
    def __init__(self, server_prop=0, inner_rounds=1, train_batch_size=4, test_batch_size=4):
        super().__init__(server_prop=server_prop, inner_rounds=inner_rounds, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        self.datasource='shakespeare'
        
        self.train_data, self.test_data = tff.simulation.datasets.shakespeare.load_data()
        self.SEQ_LENGTH = 100
        self.BATCH_SIZE = 4
        self.BUFFER_SIZE = 100
        self.TEST_BATCH_SIZE = 1

        self.vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=self.vocab, values=tf.constant(list(range(len(self.vocab))), dtype=tf.int64)), default_value=0)

        self.ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(self.vocab), mask_token=None)
        #self.chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
        #    vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

        self.list_clients = np.random.choice(self.train_data.client_ids, int(len(self.train_data.client_ids) * 0.95))
        
        if self.server_prop > 0:
            self.list_server= [i for i in self.train_data.client_ids if i not in self.list_clients]
            self.list_server = np.random.choice(self.list_server, int(len(self.list_server) * server_prop/5), replace=False)
    
    def get_length_data(self, client_id, batch_size=4):
        data = self.train_data.create_tf_dataset_for_client(client_id)
        snippets = list()
        for snippet in data:
            snippets.append(snippet['snippets'].numpy())
        snippets = b' '.join(snippets).decode('utf-8')

        return len(snippets) // 101 > batch_size

    def get_client_data(self, client_id, batch_size=4):
        #while self.get_length_data(client_id) == False:
        #    client_id = np.random.choice(self.list_clients)
        data = self.train_data.create_tf_dataset_for_client(client_id)
        return self.preprocess(data, batch_size=batch_size)

    def get_server_data(self, server_id):
        #while self.get_length_data(server_id) == False:
        #    server_id = np.random.choice(self.list_clients)
        data = self.train_data.create_tf_dataset_for_client(server_id)
        return self.preprocess(data)

    def get_test_data(self, client_id, batch_size=1):
        data = self.test_data.create_tf_dataset_for_client(client_id)
        return self.preprocess(data, batch_size=batch_size)
    
    def get_test_dataset(self, batch_size=1):
        while True:
            try:
                # n determines the number of clients to test on
                test_clients = list()
                for i in range(10):
                    cid = np.random.choice(self.list_clients)
                    while self.get_length_data(cid, 4) == False:
                        cid = np.random.choice(self.list_clients)
                    test_clients.append(cid)
                
                test_dataset = [(self.get_client_data(client, batch_size), self.get_test_data(client, batch_size)) for client in test_clients]
                return test_dataset
            except:
                continue
        
    def to_ids(self, x):
        #s = tf.reshape(x['snippets'], shape=[1])
        chars = tf.strings.bytes_split(x)
        ids = self.table.lookup(chars)
        return ids

    def preprocess(self, data, batch_size=8):
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text
        
        snippets = list()
        for snippet in data:
            snippets.append(snippet['snippets'].numpy())
        # Return client data as one long string    
        snippets = b' '.join(snippets).decode('utf-8')
        
        all_ids = self.ids_from_chars(tf.strings.unicode_split(snippets, 'UTF-8'))
        if len(all_ids) < (self.SEQ_LENGTH+1) * batch_size:
            all_ids = tf.concat([all_ids, tf.zeros((self.SEQ_LENGTH+1) * batch_size - len(all_ids), tf.int64)], 0)
        dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        dataset = dataset.batch(self.SEQ_LENGTH+1, drop_remainder=True)
        dataset = dataset.map(split_input_target)
        dataset = (dataset
            .shuffle(self.BUFFER_SIZE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        return dataset

class CIFAR100_Data(Dataset):
    def __init__(self, server_prop=0, inner_rounds=1, train_batch_size=20, test_batch_size=20):
        super().__init__()
        self.classes = 100
        self.datasource='cifar100'
        self.server_prop = server_prop

        self.train_data, self.test_data = tff.simulation.datasets.cifar100.load_data()

        self.list_clients = np.random.choice(self.train_data.client_ids, int(len(self.train_data.client_ids) * 0.95), replace=False)
        if server_prop > 0:
            self.list_server= [i for i in self.train_data.client_ids if i not in self.list_clients]
            self.list_server = np.random.choice(self.list_server, int(len(self.list_server) * server_prop/5), replace=False)
        self.list_test = self.test_data.client_ids

    def get_test_dataset(self):
        while True:
            try:
                # n determines the number of clients to test on
                test_clients = np.random.choice(self.list_clients, 10, replace=False)
                test_dataset = [self.prefetch_test_val(client) for client in test_clients]
                return test_dataset
            except:
                continue

    def prefetch_batch(self, data, batch_size=20):
        images, labels = list(), list()
        for d in data:
            image = d['image']
            label = d['label']
            images.append(image)
            labels.append(label)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(input_preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


    def prefetch_test_val(self, client_id, split=0.2):
        data = self.train_data.create_tf_dataset_for_client(client_id)
        images, labels = list(), list()
        val_images, val_labels = list(), list()
        for idx, d in enumerate(data):
            image = d['image']
            label = d['label']
            if random.random() > split:
                val_images.append(image)
                val_labels.append(label)
            else:
                images.append(image)
                labels.append(label)

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(input_preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=128, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.map(input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size=128, drop_remainder=False)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, val_dataset
        
class EMNIST_Data(Dataset):
    def __init__(self, server_prop=0):
        super().__init__()
        self.classes = 62
        self.datasource='emnist'
        self.server_prop = server_prop

        self.train_data, self.test_data = tff.simulation.datasets.emnist.load_data(only_digits=False)

        self.list_clients = np.random.choice(self.train_data.client_ids, int(len(self.train_data.client_ids) * 0.95), replace=False)
        if server_prop > 0:
            self.list_server= [i for i in self.train_data.client_ids if i not in self.list_clients]
            self.list_server = np.random.choice(self.list_server, int(len(self.list_server) * server_prop/5), replace=False)

    
    def prefetch_batch(self, data, batch_size=20):
        images, labels = list(), list()
        for d in data:
            image = d['pixels']
            label = d['label']
            images.append(image)
            labels.append(label)

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(input_preprocess_emnist, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def get_test_dataset(self):
        while True:
            try:
                # n determines the number of clients to test on
                test_clients = np.random.choice(self.list_clients, 10, replace=False)#int(len(self.list_clients) * p),replace=False)
                test_dataset = [(self.get_client_data(client), self.get_test_data(client)) for client in test_clients]
                return test_dataset
            except:
                continue

class Stackoverflow_Data(Dataset):
    def __init__(self, server_prop=0, inner_rounds=1, train_batch_size=16, test_batch_size=16):
        super().__init__(server_prop=server_prop, inner_rounds=inner_rounds, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        with open('bow.pkl', 'rb') as f:
            self.bow = pickle.load(f)
        
        logging.info('Loading Stackoverflow data from cache')
        self.train_data, _, self.test_data = tff.simulation.datasets.stackoverflow.load_data()
        self.classes=10000
        self.datasource='stackoverflow'


        self.train_data, self.test_data = create_task(list(self.bow), train_batch_size, 20, 10000)
        
        logging.info("Shuffling client ids")
        client_ids = self.train_data.client_ids
        np.random.shuffle(client_ids)
        partition = int(len(client_ids) * 0.95)
        self.list_clients = client_ids[:partition]
        if server_prop > 0:
            self.list_server= client_ids[partition:]
            #self.list_server = np.random.choice(self.list_server, int(len(self.list_server) * server_prop/5), replace=False)
            self.list_server = np.random.choice(self.list_server, server_prop * 100, replace=False)
    
    def build_to_ids_fn(self, max_sequence_length:int=20, num_out_of_vocab_buckets:int=1):
        vocab = list(self.bow)
        table_values = tf.range(len(vocab), dtype=tf.int64)
        table = tf.lookup.StaticVocabularyTable(tf.lookup.KeyValueTensorInitializer(vocab, table_values), num_oov_buckets=num_out_of_vocab_buckets)
        
        def to_ids(example):
        
            sentence = tf.reshape(example['tokens'], shape=[1])
            words = tf.strings.split(sentence, sep=' ').values
            truncated_words = words[:max_sequence_length]
            tokens = table.lookup(truncated_words) + 1
            return tokens
        
        return to_ids
    
    def batch_and_split(self, dataset: tf.data.Dataset, sequence_length:int, batch_size:int):
        def split_input_target(chunk: tf.Tensor):
            input_text = tf.map_fn(lambda x: x[:-1], chunk)
            output_text = tf.map_fn(lambda x: x[1:], chunk)
            return (input_text, output_text)
        return dataset.map(split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)#.padded_batch(batch_size, padded_shapes=[sequence_length + 1]).map(split_input_target, num_parallel_calls=tf.data.experiment.AUTOTUNE)
    
    def build_preprocess_fn(self, n=20, batch_size=16):
        shuffle_buffer_size = 1000
        num_parallel_calls = tf.data.experimental.AUTOTUNE
        
        def preprocess_fn(dataset):
            to_ids = self.build_to_ids_fn()
            dataset = dataset.shuffle(shuffle_buffer_size)
            dataset.map(to_ids, num_parallel_calls=num_parallel_calls)
            return self.batch_and_split(dataset, n, batch_size)
        
        return preprocess_fn
    
    def get_client_data(self, client_id, test=False):
        return self.train_data.create_tf_dataset_for_client(client_id)
    
    def get_server_data(self, server_id):
        return self.train_data.create_tf_dataset_for_client(server_id)

    def get_test_data(self, client_id):
        return self.test_data.create_tf_dataset_for_client(client_id)

    def get_test_dataset(self, p=0.1):
        test_clients = np.random.choice(self.list_clients, 10, replace=False)
        test_dataset = [create_p13n_data_variant(self.test_data, client) for client in test_clients]
        return test_dataset

            

def input_preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.central_crop(image, 0.75)
    return image, label

def input_preprocess_train(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.random_crop(image, (24, 24, 3))
    image = tf.image.random_flip_left_right(image)
    return image, label

def input_preprocess_emnist(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def create_p13n_data_variant(test, client_id, split:int = 1):
    data = test.create_tf_dataset_for_client(client_id)
    train_data = data.window(split, split + 1).flat_map(lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds))
    val_data = data.skip(split).window(1, split + 1).flat_map(lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds))
    return (train_data, val_data)

from typing import Tuple, List, Callable
import attr
import logging

import tensorflow_federated as tff
import tensorflow as tf


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


def create_task(vocab, batch_size:int = 16, 
                sequence_length:int = DEFAULT_SEQUENCE_LENGTH,
                vocab_size:int = DEFAULT_WORD_VOCAB_SIZE,
                num_out_of_vocab_buckets:int = 1):
    """Creates a preprocessed stackoverflow NWP task"""
    logging.info("Loading vocab dictionary from Google Storage API")
    #vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
    #vocab = list(vocab_dict.keys())[:vocab_size]

    logging.info("Initializing preprocess functions")
    preprocessing_fn = create_preprocess_fn(batch_size, vocab, sequence_length, num_out_of_vocab_buckets=num_out_of_vocab_buckets)

    logging.info("Loading dataset from cache")
    train, _, test = tff.simulation.datasets.stackoverflow.load_data()

    logging.info("Preprocessing Stackoverflow data")
    train = train.preprocess(preprocess_fn=preprocessing_fn)
    test = test.preprocess(preprocess_fn=preprocessing_fn)


    return train, test
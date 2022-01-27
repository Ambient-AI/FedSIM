import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

from typing import Dict, List, Tuple
import collections

from utils.model.models import create_cnn


def _get_emnist_datasets() -> Tuple[List[tf.data.Dataset], List[Dict[str, tf.data.Dataset]]]:
  """Pre-process EMNIST-62 dataset for FedAvg and personalization."""

  def element_fn(element):
    return (tf.expand_dims(element['pixels'], axis=-1), element['label'])

  def preprocess_train_data(dataset):
    """Pre-process the dataset for training the global model."""
    num_epochs_per_round = 10
    batch_size = 20
    buffer_size = 1000
    return dataset.repeat(num_epochs_per_round).map(element_fn).shuffle(
        buffer_size).batch(batch_size)

  def preprocess_p13n_data(dataset):

    return dataset.map(element_fn)

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=False)  # EMNIST has 3400 clients.

  # Shuffle the client ids before splitting into training and personalization.
  client_ids = list(
      np.random.RandomState(seed=42).permutation(emnist_train.client_ids))

  # The first 2500 clients are used for training a global model.
  federated_train_data = [
      preprocess_train_data(emnist_train.create_tf_dataset_for_client(c))
      for c in client_ids[:2500]
  ]

  federated_p13n_data = []
  for c in client_ids[2500:]:
    federated_p13n_data.append(
        collections.OrderedDict([
            ('train_data',
             preprocess_p13n_data(
                 emnist_train.create_tf_dataset_for_client(c))),
            ('test_data',
             preprocess_p13n_data(emnist_test.create_tf_dataset_for_client(c)))
        ]))

  return federated_train_data, federated_p13n_data

def create_task(only_digits:bool=False):
    train, test = _get_emnist_datasets()

    def model_fn():
        model = create_cnn(only_digits=only_digits)

        return tff.learning.from_keras_model(
            keras_model=model,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            input_spec=train[0].element_spec,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    return train, test, model_fn
import tensorflow as tf
import tensorflow_federated as tff

from utils.model.models import create_cnn

MAX_CLIENT_DATASET_SIZE = 3400

def _reshape(element):
    return tf.expand_dims(element['pixels'], axis=-1), element['label']

def create_preprocess_fn(batch_size:int,
                            num_parallel_calls:tf.Tensor = tf.data.experimental.AUTOTUNE):
    shuffle_buffer_size = MAX_CLIENT_DATASET_SIZE
    mapping_fn = _reshape

    def preprocess_fn(dataset):
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(mapping_fn, num_parallel_calls=num_parallel_calls)
        return dataset

    return preprocess_fn

def create_task(batch_size:int=20, only_digits:bool=False):
    preprocess_fn = create_preprocess_fn(batch_size)

    train, test = tff.simulation.datasets.emnist.load_data(only_digits=only_digits)

    train = train.preprocess(preprocess_fn)
    test = test.preprocess(preprocess_fn)
    element_spec = train.create_tf_dataset_for_client(train.client_ids[0]).element_spec


    def model_fn():
        model = create_cnn(only_digits=only_digits)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        

        return tff.learning.from_keras_model(
            keras_model=model,
            loss=loss,
            input_spec= element_spec,
            metrics=metrics
        )

    return train, test, model_fn
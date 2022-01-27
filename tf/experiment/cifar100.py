import tensorflow as tf
import tensorflow_federated as tff

from utils.models import create_resnet18

from typing import Union, Sequence, Callable, Tuple

DEFAULT_CROP_HEIGHT = 24
DEFAULT_CROP_WIDTH = 24

CIFAR_SHAPE = (32, 32, 3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
NUM_EXAMPLES_PER_CLIENT = 500

def build_image_map(
    crop_shape: Union[tf.Tensor, Sequence[int]],
    distort: bool = False
) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    if distort:
        def crop_fn(image):
            image = tf.image.random_crop(image, size=crop_shape)
            image = tf.image.random_flip_left_right(image)
            return image

    else:
        def crop_fn(image):
            return tf.image.resize_with_crop_or_pad(
                image, target_height=crop_shape[0], target_width=crop_shape[1])

    def image_map(example):
        image = tf.cast(example['image'], tf.float32)
        image = crop_fn(image)
        image = tf.image.per_image_standardization(image)
        return (image, example['label'])

    return image_map

def create_preprocess_fn(num_epochs:int, batch_size:int, 
                        crop_shape: Tuple[int, int, int] = CIFAR_SHAPE, 
                        distort:bool = False,
                        num_parallel_calls = tf.data.experimental.AUTOTUNE):
    
    shuffle_buffer_size = 1000
    image_map_fn = build_image_map(crop_shape, distort)
    def preprocess_fn(dataset):
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(image_map_fn, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size)
        return dataset
    return preprocess_fn

def create_task(num_epochs:int, batch_size:int=20, crop_height:int = DEFAULT_CROP_HEIGHT, crop_width:int = DEFAULT_CROP_WIDTH):
    crop_shape = (crop_height, crop_width, 3)
    
    # Training dataset distorts image
    train_preprocessing_fn = create_preprocess_fn(num_epochs, batch_size, crop_shape=crop_shape, distort=True)
    test_preprocessing_fn = create_preprocess_fn(num_epochs, batch_size, crop_shape=crop_shape, distort=False)
    
    train, test = tff.simulation.datasets.cifar100.load_data()

    train = train.preprocess(preprocess_fn=train_preprocessing_fn)
    test = test.preprocess(preprocess_fn=test_preprocessing_fn)

    def model_fn():
        model = create_resnet18(input_shape=crop_shape, num_classes=100)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        return model, loss, metrics
    
    return train, test, model_fn
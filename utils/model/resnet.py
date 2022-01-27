from typing import Tuple, Optional, List
import tensorflow as tf
from utils.model.group_norm import GroupNormalization
import enum

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 1e-4

class ResidualBlock(enum.Enum):
    basic = 'basic'
    bottleneck = 'bottleneck'

class NormLayer(enum.Enum):
    group_norm = 'group_norm'
    batch_norm = 'batch_norm'

def _norm_relu(input_tensor, norm):
    """Applies normalization and ReLU activation to an input tensor"""
    if tf.keras.backend.image_data_format() == 'channels_last':
        channel_axis = 3
    else:
        channel_axis = 1

    if norm is NormLayer.group_norm:
        x = GroupNormalization(axis=channel_axis)(input_tensor)
    elif norm is NormLayer.batch_norm:
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON)(
                input_tensor)
    else:
        raise ValueError('The norm argument must be of type `NormLayer`.')
    return tf.keras.layers.Activation('relu')(x)

def _conv_norm_relu(input_tensor, filters, kernel_size, norm, strides=(1, 1)):
    """Applies convolution, normalization, and ReLU activation to an input tensor"""
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
            input_tensor)
    return _norm_relu(x, norm=norm)

def _norm_relu_conv(input_tensor, filters, kernel_size, norm, strides=(1, 1)):
    """Applies normalization, ReLU activation, and convolution to an input tensor"""
    x = _norm_relu(input_tensor, norm=norm)
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x)
    return x

def _shortcut(input_tensor, residual, norm):
    """Computes the output of a shortcut block between an input and residual."""
    input_shape = tf.keras.backend.int_shape(input_tensor)
    residual_shape = tf.keras.backend.int_shape(residual)

    if tf.keras.backend.image_data_format() == 'channels_last':
        row_axis = 1
        col_axis = 2
        channel_axis = 3
    else:
        channel_axis = 1
        row_axis = 2
        col_axis = 3

    stride_width = int(round(input_shape[row_axis] / residual_shape[row_axis]))
    stride_height = int(round(input_shape[col_axis] / residual_shape[col_axis]))
    equal_channels = input_shape[channel_axis] == residual_shape[channel_axis]

    shortcut = input_tensor
    # Use a 1-by-1 kernel if the strides are greater than 1, or there the input
    # and residual tensors have different numbers of channels.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = tf.keras.layers.Conv2D(
            filters=residual_shape[channel_axis],
            kernel_size=(1, 1),
            strides=(stride_width, stride_height),
            padding='valid',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
                shortcut)

        if norm is NormLayer.group_norm:
            shortcut = GroupNormalization(axis=channel_axis)(shortcut)
        elif norm is NormLayer.batch_norm:
            shortcut = tf.keras.layers.BatchNormalization(
                axis=channel_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON)(shortcut)
        else:
            raise ValueError('The norm argument must be of type `NormLayer`.')

    return tf.keras.layers.add([shortcut, residual])

def _basic_block(input_tensor,
                 filters,
                 norm,
                 strides=(1, 1),
                 normalize_first=True):
    """Computes the forward pass of an input tensor through a basic block"""
    if normalize_first:
        x = _norm_relu_conv(
            input_tensor,
            filters=filters,
            kernel_size=(3, 3),
            strides=strides,
            norm=norm)
    else:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
                input_tensor)

    x = _norm_relu_conv(
        x, filters=filters, kernel_size=(3, 3), strides=strides, norm=norm)
    return _shortcut(input_tensor, x, norm=norm)

def _bottleneck_block(input_tensor,
                      filters,
                      norm,
                      strides=(1, 1),
                      normalize_first=True):
    """Applies a bottleneck convolutional block to a given input tensor"""
    if normalize_first:
        x = _norm_relu_conv(
            input_tensor,
            filters=filters,
            kernel_size=(1, 1),
            strides=strides,
            norm=norm)
    else:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(
                input_tensor)

    x = _norm_relu_conv(
        x, filters=filters, kernel_size=(3, 3), strides=strides, norm=norm)

    x = _norm_relu_conv(
        x, filters=filters * 4, kernel_size=(1, 1), strides=strides, norm=norm)
    return _shortcut(input_tensor, x, norm=norm)

def _residual_block(input_tensor,
                    block_function,
                    filters,
                    num_blocks,
                    norm,
                    strides=(1, 1),
                    is_first_layer=False):
    """Builds a residual block with repeating bottleneck or basic blocks."""
    x = input_tensor
    for i in range(num_blocks):
        if is_first_layer and i == 0:
            normalize_first = False
        else:
            normalize_first = True

        x = block_function(
            input_tensor=x,
            filters=filters,
            strides=strides,
            normalize_first=normalize_first,
            norm=norm)
    return x

def create_resnet(
    input_shape: Tuple[int, int, int],
    num_classes: int = 10,
    residual_block: ResidualBlock = ResidualBlock.bottleneck,
    repetitions: Optional[List[int]] = None,
    initial_filters: int = 64,
    initial_strides: Tuple[int, int] = (2, 2),
    initial_kernel_size: Tuple[int, int] = (7, 7),
    initial_max_pooling: bool = True,
    norm_layer: NormLayer = NormLayer.group_norm) -> tf.keras.Model:
    
    """Creates a ResNet v2 model with batch or group normalization.
    Instantiates the architecture from http://arxiv.org/pdf/1603.05027v2.pdf"""

    if num_classes < 1:
        raise ValueError('num_classes must be a positive integer.')

    if residual_block is ResidualBlock.basic:
        block_fn = _basic_block
    elif residual_block is ResidualBlock.bottleneck:
        block_fn = _bottleneck_block
    else:
        raise ValueError('residual_block must be of type `ResidualBlock`.')

    if not repetitions:
        repetitions = [3, 4, 6, 3]

    if initial_filters < 1:
        raise ValueError('initial_filters must be a positive integer.')

    if not isinstance(norm_layer, NormLayer):
        raise ValueError('norm_layer must be of type `NormLayer`.')

    img_input = tf.keras.layers.Input(shape=input_shape)
    x = _conv_norm_relu(
        img_input,
        filters=initial_filters,
        kernel_size=initial_kernel_size,
        strides=initial_strides,
        norm=norm_layer)

    if initial_max_pooling:
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=initial_strides, padding='same')(x)

    filters = initial_filters

    for i, r in enumerate(repetitions):
        x = _residual_block(
            x,
            block_fn,
            filters=filters,
            num_blocks=r,
            is_first_layer=(i == 0),
            norm=norm_layer)
        filters *= 2

    # Final activation in the residual blocks
    x = _norm_relu(x, norm=norm_layer)

    # Classification block
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x)

    model = tf.keras.models.Model(img_input, x)
    return model

def create_resnet18(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    norm_layer: NormLayer = NormLayer.group_norm) -> tf.keras.Model:
    return create_resnet(
        input_shape,
        num_classes,
        residual_block=ResidualBlock.basic,
        repetitions=[2, 2, 2, 2],
        norm_layer=norm_layer)

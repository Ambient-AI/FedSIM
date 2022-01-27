from tensorflow import keras
from tensorflow.keras import layers
import collections
import tensorflow_addons as tfa


def cnn(classes=62, input_dim = (28,28,1)):
    inputs = layers.Input(shape=input_dim)
    x = layers.Conv2D(32, (3, 3))(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def lstm_shakespeare(vocab_size, embedding_dim=8, lstm_units=256):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dense(vocab_size)
    ])
    return model

def lstm_stack(vocab_size, embedding_dim=96, lstm_units=670):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(lstm_units),
        layers.Dense(embedding_dim),
        layers.Dense(vocab_size, activation='softmax')
    ])
    return model


def resnet(classes=100, input_dim=(32,32,3)):
    return ResNet(
        MODELS_PARAMS['resnet18'],
        input_shape=input_dim,
        classes=classes)




def residual_conv_block(filters, stage, block, strides=(1, 1), attention=None, cut='pre'):
    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = tfa.layers.GroupNormalization(groups=2, name=bn_name + '1')(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = tfa.layers.GroupNormalization(groups=2, name=bn_name + '2')(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])
        return x

    return layer

def residual_bottleneck_block(filters, stage, block, strides=None, attention=None, cut='pre'):

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = tfa.layers.GroupNormalization(groups=2, name=bn_name + '1')(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv2D(filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = tfa.layers.GroupNormalization(groups=2, name=bn_name + '2')(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = tfa.layers.GroupNormalization(groups=2, name=bn_name + '3')(x)
        x = layers.Activation('relu', name=relu_name + '3')(x)
        x = layers.Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])

        return x

    return layer

def ResNet(model_params, input_shape=None, input_tensor=None, include_top=True,
           classes=100, **kwargs):

    img_input = layers.Input(shape=input_shape)


    ResidualBlock = model_params.residual_block
    if model_params.attention:
        Attention = model_params.attention(**kwargs)
    else:
        Attention = None

    conv_params = get_conv_params()
    init_filters = 64

    x = tfa.layers.GroupNormalization(groups=1, name='bn_data')(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = tfa.layers.GroupNormalization(groups=2, name='bn0')(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnet body
    for stage, rep in enumerate(model_params.repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            if block == 0 and stage == 0:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='post', attention=Attention)(x)

            elif block == 0:
                x = ResidualBlock(filters, stage, block, strides=(2, 2),
                                  cut='post', attention=Attention)(x)

            else:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='pre', attention=Attention)(x)

    x = tfa.layers.GroupNormalization(groups=2, name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = layers.GlobalAveragePooling2D(name='pool1')(x)
        x = layers.Dense(classes, name='fc1')(x)
        x = layers.Activation('softmax', name='softmax')(x)

    inputs = img_input

    model = keras.Model(inputs, x)

    return model



ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions', 'residual_block', 'attention']
)

MODELS_PARAMS = {
    'resnet18': ModelParams('resnet18', (2, 2, 2, 2), residual_conv_block, None),
    'resnet34': ModelParams('resnet34', (3, 4, 6, 3), residual_conv_block, None),
    'resnet50': ModelParams('resnet50', (3, 4, 6, 3), residual_bottleneck_block, None),
    'resnet101': ModelParams('resnet101', (3, 4, 23, 3), residual_bottleneck_block, None)
}


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 3
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params
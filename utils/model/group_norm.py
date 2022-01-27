import tensorflow as tf

class GroupNormalization(tf.keras.layers.Layer):
  """Group normalization layer.
    Source: 'Group Normalization' (Yuxin Wu & Kaiming He, 2018)
    https://arxiv.org/abs/1803.08494
  """

  def __init__(self,
               groups: int = 2,
               axis: int = -1,
               epsilon: float = 1e-3,
               **kwargs):
    """Constructs a Group Normalization layer"""

    super().__init__(**kwargs)
    self.supports_masking = True
    self.groups = groups
    self.axis = axis
    self.epsilon = epsilon
    self._check_axis()

  def build(self, input_shape):
    self._check_if_input_shape_is_none(input_shape)
    self._check_size_of_dimensions(input_shape)
    self._create_input_spec(input_shape)
    self.built = True
    super().build(input_shape)

  def call(self, inputs):
    """Computes the output of the layer on a given tensor."""
    input_shape = tf.keras.backend.int_shape(inputs)
    tensor_input_shape = tf.shape(inputs)
    reshaped_inputs, _ = self._reshape_into_groups(inputs, input_shape,
                                                   tensor_input_shape)
    normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

    is_instance_norm = (input_shape[self.axis] // self.groups) == 1
    if not is_instance_norm:
      outputs = tf.reshape(normalized_inputs, tensor_input_shape)
    else:
      outputs = normalized_inputs

    return outputs

  def get_config(self):
    """Returns a dictionary representing the configuration of the layer."""
    config = {
        'groups': self.groups,
        'axis': self.axis,
        'epsilon': self.epsilon,
    }
    base_config = super().get_config()
    return {**base_config, **config}

  def compute_output_shape(self, input_shape):
    return input_shape

  def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
    """Reshapes an input tensor into separate groups."""
    group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
    is_instance_norm = (input_shape[self.axis] // self.groups) == 1
    if not is_instance_norm:
      group_shape[self.axis] = input_shape[self.axis] // self.groups
      group_shape.insert(self.axis, self.groups)
      group_shape = tf.stack(group_shape)
      inputs = tf.reshape(inputs, group_shape)
    return inputs, group_shape

  def _apply_normalization(self, reshaped_inputs, input_shape):
    group_shape = tf.keras.backend.int_shape(reshaped_inputs)
    group_reduction_axes = list(range(1, len(group_shape)))
    is_instance_norm = (input_shape[self.axis] // self.groups) == 1
    if not is_instance_norm:
      axis = -2 if self.axis == -1 else self.axis - 1
    else:
      axis = -1 if self.axis == -1 else self.axis - 1
    group_reduction_axes.pop(axis)

    mean, variance = tf.nn.moments(
        reshaped_inputs, group_reduction_axes, keepdims=True)

    normalized_inputs = tf.nn.batch_normalization(
        reshaped_inputs,
        mean=mean,
        variance=variance,
        scale=None,
        offset=None,
        variance_epsilon=self.epsilon,
    )
    return normalized_inputs

  def _check_if_input_shape_is_none(self, input_shape):
    dim = input_shape[self.axis]
    if dim is None:
      raise ValueError('Axis {} of input tensor must have a defined dimension, '
                       'but the layer received an input with shape {}.'.format(
                           self.axis, input_shape))

  def _check_size_of_dimensions(self, input_shape):
    """Ensures that `input_shape` is compatible with the number of groups."""
    dim = input_shape[self.axis]
    if dim < self.groups:
      raise ValueError('Number of groups {} cannot be more than the number of '
                       'channels {}.'.format(self.groups, dim))

    if dim % self.groups != 0:
      raise ValueError('The number of channels {} must be a multiple of the '
                       'number of groups {}.'.format(dim, self.groups))

  def _check_axis(self):
    if self.axis == 0:
      raise ValueError(
          'You are trying to normalize your batch axis, axis 0, which is '
          'incompatible with GroupNorm. Consider using '
          '`tf.keras.layers.BatchNormalization` instead.')

  def _create_input_spec(self, input_shape):
    """Creates a `tf.keras.layers.InputSpec` for the GroupNorm layer."""
    dim = input_shape[self.axis]
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=len(input_shape), axes={self.axis: dim})
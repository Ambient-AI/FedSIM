import tensorflow as tf
import tensorflow_federated as tff

def subtract_weights(x, y, delta:float = 1.0):
    scaled_weights = tf.nest.map_structure(lambda x: x * delta, y)
    return tf.nest.map_structure(tf.subtract, x, scaled_weights)

@tf.function
def calculate_grads(model, dataset, phi, sum_weights):
    # Initialize model weights with phi
    optimizer = tf.keras.optimizers.SGD(learning_rate=-1)

    client_weights = model.trainable_variables
    tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, phi)

    #sum_weights = tf.nest.map_structure(lambda x: tf.Variable(tf.zeros_like(x)), phi)

    batch_idx = 0.
    for batch in dataset:
        batch_idx += 1
        with tf.GradientTape() as tape:
            output = model.forward_pass(batch)
        grads = tape.gradient(output.loss, model.trainable_variables)
        optimizer.apply_gradients(zip(tf.nest.flatten(grads), tf.nest.flatten(sum_weights)))

    return tf.nest.map_structure(lambda x: x / batch_idx, sum_weights)


def avg_weights(weight_list):
    mean_weight = list()
    for weight in zip(*weight_list):
        layer_mean = tf.math.reduce_mean(weight, axis=0)
        mean_weight.append(layer_mean)
    return mean_weight

def sum_weights(a, b):
    sum_weight = list()
    for weight in zip(*[a,b]):
        layer_sum = tf.math.reduce_sum(weight, axis=0)
        sum_weight.append(layer_sum)

    return sum_weight

def get_weights(model_fn, server_state) -> tff.learning.ModelWeights:
    model = model_fn()
    return tff.learning.ModelWeights(server_state, model.non_trainable_variables)
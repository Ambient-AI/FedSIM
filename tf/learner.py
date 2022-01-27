import tensorflow as tf

def subtract_weights(old, new, lr):
    assert(len(old) == len(new))
    new_weights = list()
    for old_weight, new_weight in zip(old, new):
        new_weights.append(tf.subtract(old_weight, tf.multiply(new_weight, lr)))
    return new_weights

@tf.function
def scale_model_weights(weights, scalar):
    weight_final = list()
    for weight in weights:
        weight_final.append(tf.multiply(weight, scalar))
    return weight_final

@tf.function
def sum_weights(weight_list):
    sum_weight = list()
    for weight in zip(*weight_list):
        layer_sum = tf.math.reduce_sum(weight, axis=0)
        sum_weight.append(layer_sum)

    return sum_weight

@tf.function    
def avg_weights(weight_list):
    mean_weight = list()
    for weight in zip(*weight_list):
        layer_mean = tf.math.reduce_mean(weight, axis=0)
        mean_weight.append(layer_mean)
    return mean_weight

def get_train_on_batch_fn():
    @tf.function
    def train_on_batch(model, loss, opt, X, y):
        with tf.GradientTape() as tape:
            y_hat = model(X, training=True)
            loss_value = loss(y, y_hat)
        grads = tape.gradient(loss_value, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
    return train_on_batch


def grads_on_batch(model, loss, X, y):
    with tf.GradientTape() as tape:
        y_hat = model(X, training=False)
        loss_value = loss(y, y_hat)
    grads = tape.gradient(loss_value, model.trainable_weights)
    return grads

def fo_grads(model, loss, data):
    grads_list = list()
    for x, y in data:
        grads_list.append(grads_on_batch(model, loss, x, y))
    
    return avg_weights(grads_list)

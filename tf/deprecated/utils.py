import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

import numpy as np
import random
import time

from scipy.stats import wasserstein_distance


def log_header(log_path, num_clients, lr, mss, comm_rounds, inner_rounds, server_prop):
    with open(f'{log_path}/{num_clients}nc_{lr}lr_{mss}ms_{comm_rounds}cr_{inner_rounds}ir_{server_prop}server_prop', 'a+') as f:
        f.write('round,acc,acc_std,loss,w_dist\n')
        f.close()

def append_logs(log_path, num_clients, lr, mss, comm_rounds, inner_rounds, server_prop, comm_round, acc, std, val_loss,w_dist):
    with open(f'{log_path}/{num_clients}nc_{lr}lr_{mss}ms_{comm_rounds}cr_{inner_rounds}ir_{server_prop}server_prop', 'a+') as f:
        f.write('{:03d},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(comm_round, acc, std, val_loss,w_dist))
        f.close()

# NEURAL NETWORK METHODS
def conv_bn(x, stride=2):
    x = layers.Conv2D(filters=64, kernel_size=3, strides=stride,padding='same')(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def build(classes=5, input_dim = (32,32,3)):
    inputs = layers.Input(shape=input_dim)
    x = conv_bn(inputs)
    x = conv_bn(x)
    x = conv_bn(x)
    x = conv_bn(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def mobilenetv2(classes=10, input_dim=(32,32,3)):
    model = keras.applications.MobileNetV2(input_shape=input_dim, classes=classes, weights=None)
    return model


##############################
# TRAINING FUNCTIONS         #
##############################

def update_weights(old_weights, grads, lr):
    new_weights = list()
    i, j = 0, 0
    while i < len(old_weights) and j < len(grads):
        if old_weights[i].shape == grads[j].shape:
            new_weights.append(old_weights[i] - lr * grads[j])
            i += 1
            j += 1
        else:
            new_weights.append(old_weights[i])
            i += 1
    return new_weights

def weight_scaling_factor(n):
    return 1/n

def weight_difference(old_weights, weights):
    new_weights = list()
    for i in range(len(weights)):
        new_weights.append(old_weights[i] - weights[i])
    return new_weights

def scale_model_weights(weights, scalar):
    weight_final = []
    steps = len(weights)
    for i in range(steps):
        weight_final.append(scalar * weights[i])
    return weight_final

def sum_weights(weight_list):
    avg_weight = list()
    for weight in zip(*weight_list):
        layer_mean = tf.math.reduce_sum(weight, axis=0)
        avg_weight.append(layer_mean)

    return avg_weight

def reduce_grads_like(grads, model):
    new_grads = list()
    i, j = 0, 0
    while i < len(grads) and j < len(model.trainable_weights):
        if  grads[i].shape == model.trainable_weights[j].shape:
            new_grads.append(grads[i])
            i += 1
            j += 1
        else:
            i += 1
    return new_grads

@tf.function
def train_on_batch(model, loss, opt, X, y):
    with tf.GradientTape() as tape:
        y_hat = model(X, training=True)
        loss_value = loss(y, y_hat)
    grads = tape.gradient(loss_value, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))

def grads(model, loss_fn, data):
    grads_list = list()
    for X, y in data:
        with tf.GradientTape() as tape:
            y_hat = model(X, training=False)
            loss_value = loss_fn(y, y_hat)
        grads = tape.gradient(loss_value, model.trainable_weights)
        grads_list.append(grads)
    fo_grads = sum_weights(grads_list)
    if len(grads_list) > 0:
        fo_grads = scale_model_weights(fo_grads, 1/len(grads_list))
    else:
        fo_grads = scale_model_weights(fo_grads, 1)
    return fo_grads


def aggregate_tensor(data):
    flatten = tf.keras.layers.Flatten()
    tensor = None
    for x, _ in data:
        if tensor != None:
            tensor = tf.math.reduce_sum(flatten(x), axis=0) + tensor
        else:
            tensor = tf.math.reduce_sum(flatten(x), axis=0)
    tensor = tf.sort(tensor)
    return tensor

def wasserstein(t1, t2):
    return wasserstein_distance(t1, t2)

##############################
# TESTING FUNCTIONS          #
##############################

def test_model(model, comm_round, start_time, dataset, test_rounds):
    old_weights = model.get_weights()
    test_data = dataset.get_test_dataset()
    losses, accs = [],[]

    for finetune_data, validation_data in test_data:
        try:
            model.fit(finetune_data, epochs=test_rounds, verbose=0)
            loss, acc = model.evaluate(validation_data, verbose=0)
            losses.append(loss)
            accs.append(acc)
        except:
            print("Error")
        model.set_weights(old_weights)
    elps = time.time() - start_time
    if len(accs) > 0:
        accs = [i for i in accs if i != None]
        losses = [i for i in losses if i != None]
        acc, std, loss = np.average(accs), np.std(accs), np.average(losses)
        print('round: {:03d} | acc= {:.4f} | std={:.4f} | loss: {:.4f} | time: {:.2f}'
                .format(comm_round, acc, std, loss, elps))
    else:
        print(f'round: {comm_round:03d} is not valid')

    return acc, std, loss

def test_model_distance(model, comm_round, start_time, dataset, test_rounds, server_tensor, mod_path=None, ckpt=False):
    if ckpt:
        assert mod_path != None
        if comm_round % 5 == 0:
            model.save(mod_path)
            print(f"Saving checkpoint model {comm_round}")
    old_weights = model.get_weights()
    test_data = dataset.get_test_dataset()
    losses, accs = [],[]
    w_distances = list()

    for finetune_data, validation_data in test_data:
        try:
            w_distances.append(wasserstein(aggregate_tensor(finetune_data), server_tensor))
            model.fit(finetune_data, epochs=test_rounds, verbose=0)
            loss, acc = model.evaluate(validation_data, verbose=0)
            losses.append(loss)
            accs.append(acc)
        except:
            print("Error")
        model.set_weights(old_weights)
    elps = time.time() - start_time
    if len(accs) > 0:
        accs = [i for i in accs if i != None]
        losses = [i for i in losses if i != None]
        acc, std, loss, dist = np.average(accs), np.std(accs), np.average(losses), np.average(w_distances)
        print('round: {:03d} | acc= {:.4f} | std={:.4f} | dist={:.4f} | loss: {:.4f} | time: {:.2f}'
                .format(comm_round, acc, std, dist, loss, elps))
    else:
        print(f'round: {comm_round:03d} is not valid')

    return acc, std, loss, dist

def evaluate_model(model, dataset, inner_batch_size, shots, classes, test_rounds, num_tests):
    old_weights = model.get_weights()
    losses = list()
    accs = list()
    for _ in range(num_tests):
        data, x_test, y_test = dataset.get_mini_dataset(inner_batch_size, 1, shots, classes, split=True)
        model.fit(data, epochs=test_rounds, verbose=0)
        evaluate = model.evaluate(x_test, y_test, verbose=0)
        losses.append(evaluate[0])
        accs.append(evaluate[1])
        model.set_weights(old_weights)
    acc, std, loss = np.average(accs), np.std(accs), np.average(losses)
    return acc, std, loss
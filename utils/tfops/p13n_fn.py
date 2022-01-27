import tensorflow_federated as tff
import tensorflow as tf

import collections

def build_personalize_fn(client_lr, batch_size:int = 20, num_epochs:int = 5, shuffle:bool=True):
    optimizer = tf.keras.optimizers.SGD(client_lr)
    @tf.function
    def personalize_fn(model: tff.learning.Model, 
                       train_data:tf.data.Dataset, 
                       test_data:tf.data.Dataset,
                       context: None):
        del context
        def train_one_batch(num_examples_sum, batch):
            # Run gradient descent on a batch
            with tf.GradientTape() as tape:
                output = model.forward_pass(batch)
            grads = tape.gradient(output.loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return num_examples_sum + output.num_examples

        training_state = 0
        metrics_dict = collections.OrderedDict()

        if shuffle:
            train_data = train_data.shuffle(1000)
        train_data = train_data.batch(batch_size)

        for idx in range(1, num_epochs + 1):
            training_state = train_data.reduce(initial_state=training_state, reduce_func=train_one_batch)
            if idx % 5 == 0 or idx == 1:
                metrics_dict[f'epoch{idx}'] = evaluate_fn(model, test_data)

        metrics_dict['num_train_samples'] = training_state
        return metrics_dict
        

    return personalize_fn

@tf.function
def evaluate_fn(model: tff.learning.Model, dataset: tf.data.Dataset):
    for var in model.local_variables:
        if var.initial_value is not None:
            var.assign(var.initial_value)
        else:
            var.assign(tf.zeros_like(var))
    
    def reduce_fn(num_examples_sum, batch):
        output = model.forward_pass(batch, training=False)
        return num_examples_sum + output.num_examples
    num_examples_sum = dataset.batch(20).reduce(initial_state=0, reduce_func=reduce_fn)
    eval_metrics = collections.OrderedDict()
    eval_metrics['num_test_examples'] = num_examples_sum
    local_outputs = model.report_local_outputs()
    for name, metric in local_outputs.items():
        if not isinstance(metric, list):
            raise TypeError(f'The metrics returned by `report_local_outputs` is '
                            f'expected to be a list, but found an instance of '
                            f'{type(metric)}. Please check that your TFF model is '
                            'built from a keras model.')
        if len(metric) == 2:
            eval_metrics[name] = metric[0] / metric[1]
        elif len(metric) == 1:
            eval_metrics[name] = metric[0]
    return eval_metrics
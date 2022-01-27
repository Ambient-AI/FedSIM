import tensorflow_federated as tff
import tensorflow as tf
import numpy as np

import collections
import functools
import logging

from utils.tfops.p13n_fn import build_personalize_fn
from utils.tfops.p13n_fn import evaluate_fn
from utils.data.data_partition import sample_train_data
from utils.data.data_partition import sample_test_data
from utils.data.data_partition import sample_test_data_variant
from utils.data.data_partition import partition_ids

class FederatedLearning:
    def __init__(self, 
            inner_epochs:int = 5,
            dataset: str = 'emnist',
            client_num_per_round: int = 10,
            test_client_num: int = 5,
            server_partition:int = 0,
            seed: int = 1111):
        tf.random.set_seed(seed)

        self.inner_epochs = inner_epochs
        self.dataset = dataset
        self.client_num = client_num_per_round
        self.test_client_num = test_client_num
        self.server_partition = server_partition
        self.cur_round = 0
        self.proxy = self.server_partition > 0
        

        logging.info(f'Initializing personalized {dataset} tasks')
        assert dataset != None

        if dataset == 'emnist':
            from experiment.emnist import create_task
            self.batch_size = 20
            self.client_lr = 0.03
            self.server_lr = 0.1

        elif dataset == 'cifar100':
            from experiment.cifar100 import create_task
            self.batch_size = 20
            self.client_lr = 0.1
            self.server_lr = 0.1

        elif dataset == 'shakespeare':
            from experiment.shakespeare import create_task
            self.batch_size = 4
            self.client_lr = 0.01
            self.server_lr = 0.1

        elif dataset == 'stackoverflow':
            from experiment.stackoverflow import create_task
            self.batch_size = 16
            self.client_lr = 0.01
            self.server_lr = 0.1
        else:
            raise ValueError(f'Expected `dataset` to be one of'
                             f'[emnist, cifar100, shakespeare, stackoverflow]. Please check '
                             f'your arguments.')

        # Initialize datasets
        self.train, self.test, self.model_fn = create_task(self.batch_size)
        self.train_ids, self.test_ids, self.proxy_ids = partition_ids(dataset, self.train, self.test)
        self.proxy_ids = self.proxy_ids[:self.server_partition]

        self.initialize_process()
        if self.proxy:
            if self.dataset == 'emnist' or self.dataset == 'shakespeare':
                self.proxy_dataset = self.train
            elif self.dataset == 'cifar100' or self.dataset == 'stackoverflow':
                self.proxy_dataset = self.test
            self.proxy_init()

    def initialize_process(self):
        # Extract appropriate federated types to decorate tff function
        logging.info('Creating `tff` models and decorating functions')
        self.whimsy_model = self.model_fn()
        self.DATA_TYPE = tff.SequenceType(self.whimsy_model.input_spec)
        self.MODEL_TYPE = self.create_server_init_fn().type_signature.result

        self.SERVER_TYPE = tff.FederatedType(self.MODEL_TYPE, tff.SERVER)
        self.CLIENT_TYPE = tff.FederatedType(self.DATA_TYPE, tff.CLIENTS)

        self.fl_process = tff.templates.IterativeProcess(
            initialize_fn= self.create_init_fn(),
            next_fn = self.create_next_fn()
        )

        logging.info('Initializing FL process')
        self.server_state = self.fl_process.initialize()

        # Build evaluation operations
        logging.info('Wrapping up personalized FL creation')
        p13n_dict = collections.OrderedDict()

        p13n_dict['fine-tune'] = functools.partial(
            build_personalize_fn,
            client_lr = self.client_lr,
            batch_size = self.batch_size
        )
        self.eval_fn = tff.learning.build_personalization_eval(
            model_fn=self.model_fn,
            personalize_fn_dict=p13n_dict,
            baseline_evaluate_fn=evaluate_fn,
            max_num_clients=10
        )

    def proxy_init(self):
        logging.info(f'Creating proxy dataset with {self.server_partition} clients and initializing model')
        proxy_data = self.proxy_dataset.from_clients_and_fn(list(self.proxy_ids), self.proxy_dataset.create_tf_dataset_for_client)
        proxy_dataset = proxy_data.create_tf_dataset_from_all_clients()

        model = self.model_fn()
        model_weights = model.trainable_variables
        opt = tf.keras.optimizers.SGD(learning_rate=self.client_lr)

        @tf.function
        def train_model(weights):
            for batch in proxy_dataset:
                with tf.GradientTape() as tape:
                    outputs = model.forward_pass(batch, training=True)
                grads = tape.gradient(outputs.loss, weights)
                opt.apply_gradients(zip(grads, weights))

            return weights
        
        model_weights = train_model(model_weights)

        logging.info('Trained initial model with proxy data')
        self.server_state, self.model =  model_weights, tff.learning.ModelWeights(model_weights, model.non_trainable_variables)

    def next(self):
        self.cur_round += 1
        if self.cur_round % 50 == 0:
            self.client_lr = self.client_lr * 0.9
        data = sample_train_data(self.train, self.train_ids, n=self.client_num)
        self.server_state, self.model = self.fl_process.next(self.server_state, data)

    def eval(self):
        if self.dataset == 'emnist' or self.dataset == 'shakespeare':
            data = sample_test_data(self.train, self.test, self.test_ids, n=self.test_client_num)
        else:
            data = sample_test_data_variant(self.test, self.test_ids, n = self.test_client_num)
        
        
        metrics = self.eval_fn(self.model, data)

        global_loss = np.array(metrics['baseline_metrics']['loss'])
        global_accs = np.array(metrics['baseline_metrics']['sparse_categorical_accuracy'])

        p13n_loss1 = np.array(metrics['fine-tune']['epoch1']['loss'])
        p13n_accs1 = np.array(metrics['fine-tune']['epoch1']['sparse_categorical_accuracy'])

        p13n_loss5 = np.array(metrics['fine-tune']['epoch5']['loss'])
        p13n_accs5 = np.array(metrics['fine-tune']['epoch5']['sparse_categorical_accuracy'])


        loss_list = [np.mean(i).item() for i in [global_loss, p13n_loss1, p13n_loss5]]
        acc_list = [np.mean(i).item() for i in [global_accs, p13n_accs1, p13n_accs5]]
        std_list = [np.std(i).item() for i in [global_accs, p13n_accs1, p13n_accs5]]

        return loss_list, acc_list, std_list


    def create_server_init_fn(self):
        # Initialize server
        @tff.tf_computation
        def server_init():
            model = self.model_fn()
            return model.trainable_variables
        return server_init

    def create_init_fn(self):
        server_init = self.create_server_init_fn()
        @tff.federated_computation
        def initialize_fn():
            return tff.federated_value(server_init(), tff.SERVER)

        return initialize_fn

    def create_client_update_fn(self):
        @tff.tf_computation(self.DATA_TYPE, self.MODEL_TYPE)
        def client_update_fn(tf_dataset, server_weights):
            model = self.model_fn()
            client_optimizer = tf.keras.optimizers.SGD(learning_rate=self.client_lr)
            return self.client_update(model, tf_dataset, server_weights, client_optimizer)

        return client_update_fn

    def create_server_update_fn(self):
        @tff.tf_computation(self.MODEL_TYPE)
        def server_update_fn(mean_client_weights):
            model = self.model_fn()
            return self.server_update(model, mean_client_weights)

        return server_update_fn

    def create_next_fn(self):
        client_update_fn = self.create_client_update_fn()
        server_update_fn = self.create_server_update_fn()

        @tff.federated_computation(self.SERVER_TYPE, self.CLIENT_TYPE)
        def next_fn(server_weights, federated_dataset):
            # Custom algorithm using TFF requires 4 main components:
            #   1. Server-to-client broadcast step
            server_weights_at_client = tff.federated_broadcast(server_weights)
            #   2. Local client update step
            client_weights = tff.federated_map(
                client_update_fn, (federated_dataset, server_weights_at_client))
            #   3. Client-to-server upload step (average the client weights)
            mean_model_deltas = tff.federated_mean(client_weights)
            #   4. Server update step
            server_weights, model = tff.federated_map(server_update_fn, mean_model_deltas)

            return server_weights, model
        
        return next_fn
        
    @tf.function
    def client_update(self, model, dataset, server_weights, client_optimizer):
        """Performs training (using the server weights) on the client's dataset"""

        client_weights = model.trainable_variables
        tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)

        for _ in range(self.inner_epochs):
            for batch in dataset:
                with tf.GradientTape() as tape:
                    outputs = model.forward_pass(batch)
                grads = tape.gradient(outputs.loss, client_weights)
                client_optimizer.apply_gradients(zip(grads, client_weights))

            return client_weights

    @tf.function
    def server_update(self, model, mean_model_deltas):
        """Updates the server model weights as  the average of the client model weights"""

        model_weights = model.trainable_variables
        tf.nest.map_structure(lambda x, y: x.assign(y), model_weights, mean_model_deltas)

        return model_weights, tff.learning.ModelWeights(model_weights, model.non_trainable_variables)

    
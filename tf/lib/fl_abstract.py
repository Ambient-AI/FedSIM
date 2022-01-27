import tensorflow as tf
import numpy as np

import logging

class FederatedLearning:
    def __init__(self,
                num_epochs:int=5,
                dataset:str = 'emnist',
                client_num_per_round:int = 10,
                server_prop:int = 5,
                lr=1e-3,
                seed=1234):
        np.random.seed(seed)
        
        self.client_num_per_round = client_num_per_round
        self.num_epochs = num_epochs
        self.lr = lr
        self.cur_round = 0

        if dataset:
            if dataset == 'emnist':
                from experiment.emnist import create_task
                # Metrics = [loss, acc]
                self.client_num_in_total = 3400
                self.test_num_in_total = 3400
            elif dataset == 'cifar100':
                from experiment.cifar100 import create_task
                # Metrics = [loss, acc]
                self.client_num_in_total = 500
                self.test_num_in_total = 100
            elif dataset == 'shakespeare':
                from experiment.shakespeare import create_task
                # Metrics = [loss, acc]
                self.client_num_in_total = 715
                self.test_num_in_total = 715
            elif dataset == 'stackoverflow':
                from experiment.stackoverflow import create_task
                # Metrics = [loss, acc, acc w/o oov, acc w/o oov or eos]
                self.client_num_in_total = 342477
                self.test_num_in_total = 204088
            else:
                raise ValueError('dataset must be one of [emnist, cifar100, shakespeare, stackoverflow]')

        # Initialize the train/test dataset and the model creation function
        logging.info(f"Creating and initializing {dataset} task")
        self.train_dataset, self.test_dataset, self.model_fn = create_task(num_epochs=num_epochs)

        # Define client ids (partition data for server-side proxy data)
        logging.info(f"Partitioning dataset")
        self.train_ids, test_ids = self.train_dataset.client_ids, self.test_dataset.client_ids
        np.random.shuffle(test_ids)
        self.test_ids = test_ids[:int(len(test_ids)*0.9)]
        self.server_ids = test_ids[int(len(test_ids)*0.9):]
        self.server_ids = np.random.choice(self.server_ids, server_prop, replace=False)

        # Initialize server model
        logging.info(f"Initializing server and client models")
        self.server_model = self.initialize_fn(initialize=server_prop > 0)
        self.client_model, self.client_loss, _ = self.model_fn()
        self.client_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=self.client_loss)


    def initialize_fn(self, initialize=False):
        """Initialize Federated Learning server-side model with fine-tuned initial weights if initialize==True"""
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        server_model, loss, metrics = self.model_fn()
        server_model.compile(optimizer=opt, loss=loss, metrics=metrics)

        if initialize:
            logging.info("Initializing server model with proxy data")
            for server_id in self.server_ids:
                dataset = self.test_dataset.create_tf_dataset_for_client(server_id)
                server_model.fit(dataset, epochs=self.num_epochs, verbose=0)
        return server_model

    def next_fn(self):
        """Executes a single training step of Federated Learning"""
        self.cur_round += 1
        logging.info(f"Training Round {self.cur_round}")

        # Broadcast the server weights to the clients
        logging.info("Broadcasting server weights to clients")
        client_list, server_weights = self.broadcast()

        # Each client computes their updated weights
        logging.info("Fine-tuning client models using private datasets")
        client_weights = [self.client_update(
                            self.train_dataset.create_tf_dataset_for_client(client), 
                            server_weights
                        ) for client in client_list]

        # The server computes updated parameters using these updates
        logging.info("FedAveraging client weights")
        mean_client_weights = self.server_compute(client_weights)

        # The server updates its model
        logging.info("Updating server model")
        server_weights = self.server_update(mean_client_weights)


        logging.info("Evaluating model performance")
        
        tf.keras.backend.clear_session()

    def broadcast(self):
        return list(np.random.choice(self.train_ids, 
                                    self.client_num_per_round, 
                                    replace=False)
                    ), self.server_model.get_weights()

    def client_update(self, dataset, server_weights):
        # Initialize the client model with the current server weights
        self.client_model.set_weights(server_weights)
        self.client_model.fit(dataset, epochs=self.num_epochs, verbose=0)
        return self.client_model.get_weights()


    @tf.function
    def server_compute(self, client_weights):
        # FedAvg weights list
        avg_weight = list()
        for weight in zip(*client_weights):
            layer_mean = tf.math.reduce_mean(weight, axis=0)
            avg_weight.append(layer_mean)

        return avg_weight

    def server_update(self, mean_client_weights):
        self.server_model.set_weights(mean_client_weights)
        return mean_client_weights

    def server_evaluate(self, num_clients:int=1):
        server_weights = self.server_model.get_weights()

        metrics_list = list()

        for _ in range(num_clients):
            client_id = np.random.choice(self.test_ids)
            train_dataset = self.train_dataset.create_tf_dataset_for_client(client_id)
            test_dataset = self.test_dataset.create_tf_dataset_for_client(client_id)

            # Fine-tune on client training data (meta-training)
            self.server_model.fit(train_dataset, epochs=self.num_epochs, verbose=0)

            # Evaluate on client testing data
            metrics = self.server_model.evaluate(test_dataset, verbose=0)
            metrics_list.append(metrics)

            # Revert server_model to original weights
            self.server_model.set_weights(server_weights)

        return np.average(metrics_list, axis=0)
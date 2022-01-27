import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import numpy as np
import time
import pickle

import utils
import models

class hf2_meta:
    def __init__(self, model, dataset, meta_step_size, learning_rate, 
                inner_rounds, comm_rounds, test_rounds, num_clients,
                shots, classes, server_data: bool, use_clients: bool, 
                verbose:bool, input_dim=(32,32,3), logdir='', moddir='', ckpt=False):
        self.model = model
        self.data = dataset
        self.mss = meta_step_size
        self.lr = learning_rate
        self.inner_rounds = inner_rounds
        self.comm_rounds = comm_rounds
        self.test_rounds = test_rounds
        self.num_clients = num_clients
        self.use_server_data = server_data
        self.use_clients = use_clients
        self.verbose = verbose
        self.inner_batch_size = shots * classes
        self.classes = classes
        self.shots = shots
        self.input_dim=input_dim
        self.logdir = logdir
        self.moddir = moddir
        self.ckpt = ckpt

        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
        
        # Training hyperparameters
        self.lambda_reg = 1.0
        self.delta = 0.25
        self.inner_lr = 0.1

        

        if self.data.datasource == 'omniglot' or self.data.datasource == 'fashion_mnist' or self.data.datasource == 'emnist' or self.data.datasource == 'mnist':
            self.global_model = models.cnn(classes=dataset.classes, input_dim=input_dim)
            self.upper_model = models.cnn(classes=dataset.classes, input_dim=input_dim)
            self.lower_model = models.cnn(classes=dataset.classes, input_dim=input_dim)
            self.optimizer = keras.optimizers.Adam()
            self.use_logits = False
            
        elif self.data.datasource == 'cifar100':
            self.global_model = models.resnet(classes=self.data.classes, input_dim=self.input_dim)
            self.upper_model = models.resnet(classes=self.data.classes, input_dim=self.input_dim)
            self.lower_model = models.resnet(classes=self.data.classes, input_dim=self.input_dim)
            self.optimizer = keras.optimizers.Adam()
            self.use_logits = False

        elif self.data.datasource == 'shakespeare':
            self.global_model = models.lstm_shakespeare(len(self.data.vocab))
            self.upper_model = models.lstm_shakespeare(len(self.data.vocab))
            self.lower_model = models.lstm_shakespeare(len(self.data.vocab))
            self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
            self.use_logits = True
        
        elif self.data.datasource == 'stackoverflow':
            self.global_model = models.lstm_stack(self.data.classes)
            self.upper_model = models.lstm_stack(self.data.classes)
            self.lower_model = models.lstm_stack(self.data.classes)
            self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
            self.use_logits = False
        # Create local model for clients
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)
        # Create a global model deployed on the server
        self.global_model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)
        self.upper_model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)
        self.lower_model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)


        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=self.use_logits)

        # Add l2 regularization to the model
        def imaml_regularizer():
            reg = tf.add_n([ tf.reduce_sum(tf.square(self.model.get_weights[i] - self.global_model.get_weights[i]))
                for i in range(len(self.model.get_weights()))])
            return 0.5 * self.lambda_reg * reg

        self.model.add_loss(imaml_regularizer)

        self.server_tensor = None

        if self.ckpt:
            try:
                model = tf.keras.models.load_model(moddir)
                print("Starting from checkpoint")
            except:
                print("Starting from scratch")


    def train(self):
        global_acc = list()
        global_loss = list()

        if self.verbose:
            utils.log_header(self.logdir, self.num_clients, self.lr, self.mss, self.comm_rounds, self.inner_rounds, self.data.get_server_proportion())

        if self.use_server_data:
            for client in self.data.list_server:
                server_data = self.data.get_server_data(client)
                weights = self.train_client(self.model, server_data)
                self.global_model.set_weights(weights)

            if self.server_tensor != None:
                self.server_tensor = utils.aggregate_tensor(server_data) + self.server_tensor
            else:
                self.server_tensor = utils.aggregate_tensor(server_data)

            acc, std, val_loss = utils.test_model(self.global_model, 0, time.time(), self.data, self.test_rounds)

        for comm_round in range(self.comm_rounds):
            local_weights_list = list()
            final_grads_list = list()

            global_weights = self.global_model.get_weights()

            cur_meta_step_size = (1-(comm_round/self.comm_rounds)) * self.mss

            start = time.time()
            ### Client-Side Calculations ###
            for idx, client in enumerate(np.random.choice(self.data.list_clients, size=self.num_clients, replace=False)):
                self.model.set_weights(global_weights)
                
                data = self.data.get_client_data(client)

                weights = self.train_client(self.model, data)

                local_weights_list.append(weights)

                K.clear_session()

            ### Server-Side Calculations ###           

            for idx, phi_k in enumerate(local_weights_list):
                v_k = utils.weight_difference(phi_k, global_weights)
                data_query = self.data.get_server_data(np.random.choice(self.data.list_server))

                try:
                    Hv = self.hessian_free(self.global_model, v_k, data_query, self.delta)

                    final_grads = utils.update_weights(v_k, Hv, self.lr)
                    final_weights = utils.update_weights(phi_k, final_grads, cur_meta_step_size)

                except:
                    final_weights = utils.update_weights(phi_k, v_k, cur_meta_step_size)

                final_grads_list.append(final_weights)
                
            
            # Average the gradients and update global parameters
            final_weights = utils.scale_model_weights(utils.sum_weights(final_grads_list), 
                                                utils.weight_scaling_factor(self.num_clients))

            self.global_model.set_weights(final_weights)

            
            acc, std, val_loss, wdist = utils.test_model_distance(self.global_model, comm_round+1, start, self.data, self.test_rounds, self.server_tensor, self.moddir, self.ckpt)

            if self.verbose:
                utils.append_logs(self.logdir, self.num_clients, self.lr, self.mss, self.comm_rounds, self.inner_rounds, self.data.get_server_proportion(), comm_round, acc, std, val_loss, wdist)


            global_acc.append(acc)
            global_loss.append(val_loss)

            K.clear_session()

        return self.global_model, global_acc, global_loss
    
    def train_client(self, model, data): 
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=self.use_logits)
        
        for epoch in range(self.inner_rounds):
            for step, (x_batch_train, y_batch_train) in enumerate(data):
                utils.train_on_batch(model, loss_fn, self.optimizer, x_batch_train, y_batch_train)

        return model.get_weights()
    
    def hessian_free(self, model, grads, data, delta):
        # Based on the Lipschitz continutity of the Hessian of phi

        # Calculate theta +- first order grads
        upper_weights = utils.update_weights(model.get_weights(), utils.scale_model_weights(grads, -1), delta)
        lower_weights = utils.update_weights(model.get_weights(), grads, delta)

        self.upper_model.set_weights(upper_weights)
        self.lower_model.set_weights(lower_weights)

        grads_upper = utils.grads(self.upper_model, self.loss_fn, data)
        grads_lower = utils.grads(self.lower_model, self.loss_fn, data)

        hessian_grads = utils.weight_difference(grads_upper, grads_lower)
        hessian_grads = utils.scale_model_weights(hessian_grads, 1/(2 * delta))

        return hessian_grads


    def save_accs(self, accs, filename):
        with open(filename, 'wb') as f:
            pickle.dump(accs, f)

    def save_loss(self, loss, filename):
        with open(filename, 'wb') as f:
            pickle.dump(loss, f)

    def save_weights(self, model, filename):
        model.save(filename)
import tensorflow as tf
import tensorflow_federated as tff

from lib.fl_abstract import FederatedLearning

from utils.data.data_partition import sample_train_data
from utils.tfops.learner import calculate_grads, subtract_weights

class FedSim(FederatedLearning):
    def __init__(self, inner_epochs: int = 5, dataset: str = 'emnist', 
                client_num_per_round: int = 10,
                test_client_num: int = 5, server_partition: int = 5, seed: int = 1111,
                alpha_lr: float=0.1, beta_lr: float = 0.1, delta: float = 0.1, lambda_reg: float = 1.0):

        self.alpha_lr = alpha_lr
        self.beta_lr = beta_lr
        self.delta = delta
        self.lambda_reg = lambda_reg

        super().__init__(inner_epochs=inner_epochs, dataset=dataset, client_num_per_round=client_num_per_round, test_client_num=test_client_num, server_partition=server_partition, seed=seed)

        if server_partition <= 0:
            raise ValueError(f"The value of `server_partition` must be a positive integer "
                             f"(currently {server_partition}.\nPlease check the init values")

        

    def next(self):
        self.cur_round += 1
        if self.cur_round % 50 == 0:
            self.client_lr = self.client_lr * 0.9
        data = sample_train_data(self.train, self.train_ids, n=self.client_num)
        proxy_data = sample_train_data(self.train, self.proxy_ids, n=self.client_num)
        self.server_state, self.model = self.fl_process.next(self.server_state, data, proxy_data)
        
    def create_server_update_fn(self):
        @tff.tf_computation(self.MODEL_TYPE, self.MODEL_TYPE)
        def server_update_fn(server_weights, mean_client_deltas):
            model = self.model_fn()
            return self.server_update(model, server_weights, mean_client_deltas)

        return server_update_fn

    def create_client_update_fn(self):
        @tff.tf_computation(self.DATA_TYPE, self.DATA_TYPE, self.MODEL_TYPE)
        def client_update_fn(tf_dataset, proxy_dataset, server_weights):
            model = self.model_fn()
            client_optimizer = tf.keras.optimizers.SGD(learning_rate=self.client_lr)
            phi = self.client_update(model, tf_dataset, server_weights, client_optimizer)

            # Simulate Hessian calculation in server
            # Hessian = (∇(θ + δ*vₖ) - ∇(θ - δ*vₖ))/ 2δ
            hessian_grads = self.hessian_calc(model, proxy_dataset, phi, server_weights)

            # gₖ = vₖ - β*hₖ
            g_k = subtract_weights(phi, hessian_grads, self.beta_lr)


            return g_k
        return client_update_fn
    
    def create_next_fn(self):
        client_update_fn = self.create_client_update_fn()
        server_update_fn = self.create_server_update_fn()

        @tff.federated_computation(self.SERVER_TYPE, self.CLIENT_TYPE, self.CLIENT_TYPE)
        def next_fn(server_weights, federated_dataset, proxy_dataset):
            server_weights_at_client = tff.federated_broadcast(server_weights)

            grads = tff.federated_map(
                client_update_fn, (federated_dataset, proxy_dataset, server_weights_at_client))

            mean_model_deltas = tff.federated_mean(grads)

            server_weights, model = tff.federated_map(server_update_fn, (server_weights, mean_model_deltas))

            return server_weights, model
        
        return next_fn


    def hessian_calc(self, model, proxy, phi, server_weights):
        # tf.2.x does not support multiple variable calls -> initialize at the beginning
        upper_temp = tf.nest.map_structure(lambda x: tf.Variable(tf.zeros_like(x)), phi)
        lower_temp = tf.nest.map_structure(lambda x: tf.Variable(tf.zeros_like(x)), phi)

        # Calculate θ+δv and θ-δv
        upper_weights = subtract_weights(server_weights, phi, -self.delta)
        lower_weights = subtract_weights(server_weights, phi, self.delta)

        # Calcuate ∇(θ+δv) and ∇(θ-δv) w.r.t. proxy data
        upper_grads = calculate_grads(model, proxy, upper_weights, upper_temp)
        lower_grads = calculate_grads(model, proxy, lower_weights, lower_temp)

        # Hessian = [∇(θ+δv)-∇(θ-δv)]/2δ
        hessian_grads = tf.nest.map_structure(tf.subtract, upper_grads, lower_grads)
        hessian_grads = tf.nest.map_structure(lambda x: x / (2*self.delta), hessian_grads)

        return hessian_grads

    @tf.function
    def client_update(self, model, dataset, server_weights, client_optimizer):
        """Performs training (using the server weights) on the client's dataset"""

        client_weights = model.trainable_variables
        tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)

        for _ in range(self.inner_epochs):
            for batch in dataset:
                with tf.GradientTape() as tape:
                    outputs = model.forward_pass(batch)
                    reg_loss = self.imaml_regularizer(client_weights, server_weights, outputs.loss)
                grads = tape.gradient(reg_loss, client_weights)
                client_optimizer.apply_gradients(zip(grads, client_weights))

        return tf.nest.map_structure(tf.subtract, client_weights, server_weights)

    def imaml_regularizer(self, client, server, loss):
        reg = tf.add_n([ tf.reduce_sum(tf.square(client[i] - server[i]))
                for i in range(len(client))])

        return loss + 0.5 * self.lambda_reg * reg

    @tf.function
    def server_update(self, model, server_weights, mean_client_deltas):
        """Updates the server model weights as  the average of the client model weights"""

        new_weights = tf.nest.map_structure(lambda x,y: x + y * self.server_lr, server_weights, mean_client_deltas)

        return new_weights, tff.learning.ModelWeights(new_weights, model.non_trainable_variables)

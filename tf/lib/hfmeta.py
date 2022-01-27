import tensorflow as tf
import numpy as np

from lib.fl_abstract import FederatedLearning
from utils.learner import avg_weights, scale_model_weights, subtract_weights, fo_grads

class HFMeta(FederatedLearning):
    def __init__(self, num_epochs: int=5, dataset: str = 'emnist', 
                client_num_per_round: int = 10, server_prop: int = 5, 
                lr=0.001, seed=1234,
                meta_step_size:float=0.01,
                lambda_reg:float=1.0,
                delta:float=0.1):
        super().__init__(num_epochs, dataset=dataset, 
                        client_num_per_round=client_num_per_round, 
                        server_prop=server_prop, lr=lr, seed=seed)
        
        self.meta_step_size = meta_step_size

        self.lambda_reg = lambda_reg
        self.delta = delta

        self.upper_model, self.upper_loss, _ = self.model_fn()
        self.lower_model, self.lower_loss, _ = self.model_fn()

    
    def client_update(self, dataset, server_weights):
        # TODO: Add l2 regularization to the model

        # Initialize the client model with the current server weights
        model, loss, _ = self.model_fn()
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.set_weights(server_weights)
        model.compile(optimizer=opt, loss=loss)
        model.fit(dataset, epochs=self.num_epochs, verbose=0)
        return model.get_weights()

    def server_compute(self, client_weights):
        weights_list = list()
        server_weights = self.server_model.get_weights()
        for phi_k in client_weights:
            v_k = subtract_weights(phi_k, server_weights, 1)
            data_query = self.test_dataset.create_tf_dataset_for_client(np.random.choice(self.server_ids))

            # Hessian = (∇(θ + δ*vₖ) - ∇(θ - δ*vₖ))/ 2δ
            h_k = self.hessian_free(v_k, data_query, self.delta)

            # gₖ = vₖ - α*hₖ
            g_k = subtract_weights(v_k, h_k, self.lr)

            # θₖ₊₁ = θₖ - β*gₖ 
            theta_new = subtract_weights(server_weights, g_k, self.meta_step_size)

            # Append θₖ₊₁ to the weights list
            weights_list.append(theta_new)
        return avg_weights(weights_list)

    def hessian_free(self, grads, data, delta):
        server_weights = self.server_model.get_weights()
        # Calculate θ+δv and θ-δv
        upper_weights = subtract_weights(server_weights, grads, -delta)
        lower_weights = subtract_weights(server_weights, grads, delta)

        self.upper_model.set_weights(upper_weights)
        self.lower_model.set_weights(lower_weights)

        # Calcuate ∇(θ+δv) and ∇(θ-δv)
        upper_grads = fo_grads(self.upper_model, self.upper_loss, data)
        lower_grads = fo_grads(self.lower_model, self.lower_loss, data)

        # Hessian = [∇(θ+δv)-∇(θ-δv)]/2δ 
        hessian_grads = subtract_weights(upper_grads, lower_grads, 1)
        hessian_grads = scale_model_weights(hessian_grads, 1/(2*delta))

        return hessian_grads
import logging
import argparse
import wandb


def add_args(parser):
    """
    parser: argparse.ArggumentParser
    return a parser with args required for experiments
    """

    # Training settings
    parser.add_argument('--method', type=str, default='fed', help='methodology to train model')
    parser.add_argument('--dataset', type=str, default='emnist',  help='dataset used for training')
    parser.add_argument('--client_num_in_total', type=int, default=1000,  help='number of workers in a distributed cluster')
    parser.add_argument('--client_num_per_round', type=int, default=10, help='number of workers')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default 0.001)')
    parser.add_argument('--meta_step_size', type=float, default=0.1, help='meta training learning rate (default=0.01)')
    parser.add_argument('--lambda_reg', type=float, default=1.0, help='lambda regularizer for hf2meta')
    parser.add_argument('--delta', type=float, default=0.1, help='delta for hessian grads')
    parser.add_argument('--epochs', type=int, default=5, help='how many epochs will be trained locally')
    parser.add_argument('--comm_round', type=int, default=10, help='how many communication rounds')
    
    # Testing settings
    parser.add_argument('--test_client', type=int, default=10, help='number of test clients')
    parser.add_argument('--server_partition', type=int, default=0, help='partitioned data in server')

    # Environment settings
    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    # Logging settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level (0-2)')

    args = parser.parse_args()
    return args


def init_method(args):
    method_name= args.method
    logging.info(f"initializing methodology. method name={method_name}")

    if method_name=='hfmeta':
        from lib.hfmeta import HFMeta
        method = HFMeta(num_epochs=args.epochs, dataset=args.dataset,client_num_per_round=args.client_num_per_round, server_prop=args.server_partition, lr=args.lr, seed=args.seed, meta_step_size=args.meta_step_size, lambda_reg=args.lambda_reg, delta=args.delta)

    elif method_name == 'jiang':
        from lib.reptile import Reptile
        method = Reptile(num_epochs=args.epochs, dataset=args.dataset,client_num_per_round=args.client_num_per_round, server_prop=args.server_partition, lr=args.lr, seed=args.seed, meta_step_size=args.meta_step_size)

    else: # Default case is federated
        from lib.fl_abstract import FederatedLearning
        method = FederatedLearning(num_epochs=args.epochs, dataset=args.dataset,client_num_per_round=args.client_num_per_round, server_prop=args.server_partition, lr=args.lr, seed=args.seed)

    return method


def main():
    args = add_args(argparse.ArgumentParser())

    if args.verbose == 0:
        pass
    elif args.verbose == 1:
        logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    elif args.verbose == 2:
        pass

    config = {
        "method": args.method,
        "dataset": args.dataset,
        "comm_rounds": args.comm_round,
        "inner_epochs": args.epochs,
        "lr": args.lr,
        "meta_step_size": args.meta_step_size,
        "server_partition": args.server_partition
    }

    logging.info("Initializing wandb session")
    wandb.init(project='hfmeta', entity='steve2972', config=config)

    logging.info("Initializing methodology")
    method = init_method(args)

    for _ in range(args.comm_round):
        method.next_fn()
        metrics = method.server_evaluate(args.test_client)
        logging.info(f"Test loss: {metrics[0]:.4f} | Test Accuracy: {metrics[1]:.4f}")
        log = {
            "loss": metrics[0],
            "accuracy": metrics[1],
            "comm_round": method.cur_round
        }
        wandb.log(log)

if __name__ == '__main__':
    main()
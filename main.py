import numpy as np
import yaml
import logging
import time
import nest_asyncio
import argparse
import wandb

def parse_args(parser, yaml_path:str = 'config-defaults.yaml'):
    parser.add_argument('--server-partition', type=int, default=0, help='partitioned data in server')
    parsed_args = parser.parse_args()
    with open(yaml_path, 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    if int(args['server_partition']['value']) != parsed_args.server_partition:
        args['server_partition']['value'] = parsed_args.server_partition
        with open(yaml_path, 'w') as stream:
            yaml.dump(args, stream)
    return args


def init_method(args):
    method_name= args['method']['value']
    logging.info(f"initializing methodology. method name={method_name}")

    if method_name=='fedsim':
        from lib.fedsim import FedSim
        method = FedSim(inner_epochs = args['inner_epochs']['value'], dataset = args['dataset']['value'], 
                client_num_per_round=args['client_num_per_round']['value'], test_client_num=args['test_client_num']['value'], 
                server_partition=args['server_partition']['value'], seed=args['random_seed']['value'], alpha_lr=args['alpha']['value'],
                beta_lr=args['beta']['value'], delta=args['delta']['value'], lambda_reg=args['lambda_reg']['value'])

    elif method_name == 'jiang':
        from lib.reptile import Reptile
        method = Reptile(inner_epochs = args['inner_epochs']['value'], dataset = args['dataset']['value'], 
                client_num_per_round=args['client_num_per_round']['value'], test_client_num=args['test_client_num']['value'], 
                server_partition=args['server_partition']['value'], seed=args['random_seed']['value'], meta_step_size=args['beta']['value'])

    elif method_name == 'fedprox':
        from lib.fedprox import FedProx
        method = FedProx(inner_epochs = args['inner_epochs']['value'], dataset = args['dataset']['value'], 
                client_num_per_round=args['client_num_per_round']['value'], test_client_num=args['test_client_num']['value'], 
                server_partition=args['server_partition']['value'], seed=args['random_seed']['value'])

    elif method_name == 'pfedme':
        from lib.pfedme import PFedMe
        method = PFedMe(inner_epochs = args['inner_epochs']['value'], dataset = args['dataset']['value'], 
                client_num_per_round=args['client_num_per_round']['value'], test_client_num=args['test_client_num']['value'], 
                server_partition=args['server_partition']['value'], seed=args['random_seed']['value'])

    else: # Default case is federated
        from lib.fl_abstract import FederatedLearning as FL
        method = FL(inner_epochs = args['inner_epochs']['value'], dataset = args['dataset']['value'], 
                client_num_per_round=args['client_num_per_round']['value'], test_client_num=args['test_client_num']['value'], 
                server_partition=args['server_partition']['value'], seed=args['random_seed']['value'])

    return method
    


def main():
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%y-%d-%b %H:%M:%S', level=logging.INFO)
    args = parse_args(argparse.ArgumentParser())

    
    logging.info("Initializing wandb session")
    wandb.init(project='HF2-Meta', entity='steve2972')

    nest_asyncio.apply()

    logging.info("Initializing methodology")
    method = init_method(args)
    wandb.config.update({"Proxy IDs":method.proxy_ids})

    best_global_acc, best_p13n_acc_1, best_p13n_acc_5 = list(), list(), list()
    def update_best(acc_list, new_acc, name='', best_of:int = 10):
        if len(acc_list) < best_of:
            acc_list.append(new_acc)

        else:
            if new_acc > np.min(acc_list):
                acc_list[np.argmin(acc_list)] = new_acc
                wandb.run.summary[name] = np.mean(acc_list)
                wandb.run.summary[f'{name}_std'] = np.std(acc_list)
                return acc_list
        return acc_list


    for _ in range(args['comm_rounds']['value']):
        t1 = time.time()
        method.next()    # Run 1 communication round
        t2 = time.time()
        losses, accs, stds = method.eval()    # (Baseline, 1 round fine-tuning, 5 rounds fine-tuning) x 2
        t3 = time.time()

        logging.info(f"Round {method.cur_round} | time train: {t2-t1:.3f}s, eval: {t3-t2:.3f}s | p13n loss: {losses[2]:.3f} p13n acc: {accs[2]:.3f}")
        
        log = {
            "global_loss": losses[0],
            "p13n_loss_1": losses[1],
            "p13n_loss_5": losses[2],
            "global_acc": accs[0],
            "p13n_acc_1": accs[1],
            "p13n_acc_5": accs[2],
            "global_acc_std": stds[0],
            "p13n_acc_1_std": stds[1],
            "p13n_acc_5_std": stds[2],
            "comm_round": method.cur_round
        }
        wandb.log(log)

        best_global_acc = update_best(best_global_acc, accs[0], 'best_global_acc')
        best_p13n_acc_1 = update_best(best_p13n_acc_1, accs[1], 'best_p13n_acc_1')
        best_p13n_acc_5 = update_best(best_p13n_acc_5, accs[2], 'best_p13n_acc_5')


if __name__ == '__main__':
    main()
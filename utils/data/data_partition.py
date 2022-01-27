import numpy as np
import tensorflow as tf

import collections

def create_p13n_data(train, test, client_id):
    def unbatch(dataset):
        return dataset.unbatch()

    train = train.preprocess(unbatch)
    test = test.preprocess(unbatch)

    return collections.OrderedDict([
        ('train_data', train.create_tf_dataset_for_client(client_id)),
        ('test_data', test.create_tf_dataset_for_client(client_id))
    ])

def create_p13n_data_variant(test, client_id, split:int = 1):
    def unbatch(dataset):
        return dataset.unbatch()
    test = test.preprocess(unbatch)

    data = test.create_tf_dataset_for_client(client_id)
    train_data = data.window(split, split + 1).flat_map(lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds))
    val_data = data.skip(split).window(1, split + 1).flat_map(lambda *ds: ds[0] if len(ds) == 1 else tf.data.Dataset.zip(ds))
    return collections.OrderedDict([
        ('train_data', train_data),
        ('test_data', val_data)
    ])

def sample_train_data(data, client_ids, n:int=10):
    return [data.create_tf_dataset_for_client(cid) for cid in np.random.choice(client_ids, n, replace=True)]

def sample_test_data(train, test, client_ids, n:int=10):
    data_idxs = list(np.random.choice(client_ids, n, replace=False))
    data = [create_p13n_data(train, test, cid) for cid in data_idxs]
    return data

def sample_test_data_variant(test, client_ids, n:int=10):
    data_idxs = list(np.random.choice(client_ids, n, replace=False))
    data = [create_p13n_data_variant(test, cid) for cid in data_idxs]
    return data

def sample_proxy_data(proxy, client_ids, n:int=10):
    return [proxy.create_tf_dataset_for_client(cid) for cid in np.random.choice(client_ids, n, replace=False)]


def partition_ids(dataset_name:str, train, test):
    train_ids = train.client_ids
    test_ids = test.client_ids

    if dataset_name == 'emnist' or dataset_name == 'shakespeare':
        client_ids = np.random.permutation(train_ids)

        if dataset_name == 'emnist':
            train_ids = client_ids[:2380]
            test_ids = client_ids[2380:3060]
            proxy_ids = client_ids[3060:]
        
        else:
            train_ids = client_ids[:500]
            test_ids = client_ids[500:645]
            proxy_ids = client_ids[645:]

    elif dataset_name == 'cifar100':
        client_ids = np.random.permutation(test_ids)
        proxy_ids = client_ids[:50]
        test_ids = client_ids[50:]

    elif dataset_name == 'stackoverflow':
        test_ids = np.intersect1d(train_ids, test_ids)
        train_ids = np.setdiff1d(train_ids, test_ids)

        test_ids = np.random.permutation(test_ids)
        proxy_ids = test_ids[-1000:]
        test_ids = test_ids[:1000]

    return train_ids, test_ids, proxy_ids
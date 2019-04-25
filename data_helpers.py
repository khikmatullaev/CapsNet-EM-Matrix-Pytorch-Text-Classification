import numpy as np
import torch
import torch.utils.data as utils


def load_dataset(dataset_name, batch_size, type, folder = './datasets_prepared'):
    torch.cuda.empty_cache()

    dataset_size = {
        "IMDB": 2,
        "ProcCons": 2,
        'MR':2,
        'SST-1':5,
        'SST-2':2,
        'SUBJ':2,
        'TREC':6
    }

    params = {'batch_size': batch_size,
              'drop_last': True,
              'shuffle': True,
              'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    X = np.load(folder + '/' + dataset_name + '/' + type + '/X.npy')
    Y = np.load(folder + '/' + dataset_name + '/' + type + '/Y.npy')

    X_train = X[:X.shape[0] * 70 // 100]
    Y_train = Y[:Y.shape[0] * 70 // 100]

    train_x = torch.stack([torch.Tensor(i) for i in X_train])  # transform to torch tensors
    train_y = torch.from_numpy(Y_train)

    train = utils.TensorDataset(train_x, train_y)
    train_dataloader = utils.DataLoader(train, **params)

    X_dev = X[X.shape[0] * 70 // 100:X.shape[0] * 85 // 100]
    Y_dev = Y[Y.shape[0] * 70 // 100:Y.shape[0] * 85 // 100]

    dev_x = torch.stack([torch.Tensor(i) for i in X_dev])  # transform to torch tensors
    dev_y = torch.from_numpy(Y_dev)

    dev = utils.TensorDataset(dev_x, dev_y)
    dev_dataloader = utils.DataLoader(dev, **params)

    X_test = X[X.shape[0] * 85 // 100:]
    Y_test = Y[Y.shape[0] * 85 // 100:]

    test_x = torch.stack([torch.Tensor(i) for i in X_test])  # transform to torch tensors
    test_y = torch.from_numpy(Y_test)

    test = utils.TensorDataset(test_x, test_y)
    test_dataloader = utils.DataLoader(test, **params)

    return train_dataloader, dev_dataloader, test_dataloader, dataset_size[dataset_name]
import numpy as np
import torch.utils.data as utils
import torch


def load_dataset(dataset_name, type, folder = './datasets_prepared'):
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

    X = np.load(folder + '/' + dataset_name + '/' + type + '/X.npy')
    Y = np.load(folder + '/' + dataset_name + '/' + type + '/Y.npy')

    X_train = X[:X.shape[0] * 85 // 100]
    Y_train = Y[:Y.shape[0] * 85 // 100]

    train_x = torch.stack([torch.Tensor(i) for i in X_train])  # transform to torch tensors
    train_y = torch.stack([torch.Tensor(i) for i in Y_train])
    train = utils.TensorDataset(train_x, train_y)
    train_dataloader = utils.DataLoader(train, batch_size=128, shuffle=True, **kwargs)

    X_test = X[X.shape[0] * 85 // 100:]
    Y_test = Y[Y.shape[0] * 85 // 100:]

    test_x = torch.stack([torch.Tensor(i) for i in X_test])  # transform to torch tensors
    test_y = torch.stack([torch.Tensor(i) for i in Y_test])
    test = utils.TensorDataset(test_x, test_y)
    test_dataloader = utils.DataLoader(test, batch_size=128, shuffle=True, **kwargs)

    return train_dataloader, test_dataloader, Y.shape[1]

# load_dataset('MR', 'fasttext')
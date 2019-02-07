import time
import os
import torch.optim as optim

from data_helpers import load_dataset
from capsules import *
from spread_loss import *


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def snapshot(model, folder, epoch):
    path = os.path.join(folder, 'model_{}.pth'.format(epoch))

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        r = (1. * batch_idx + (epoch - 1) * train_len) / (args.epochs * train_len)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        epoch_acc += acc[0].item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tAccuracy: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item(), acc[0].item(),
                batch_time=batch_time, data_time=data_time))
    return epoch_acc


def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()

    test_loss /= test_len
    acc /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    return acc


def main():
    em_types = ['fasttext', 'glove', 'word2vec']
    databases = ['CR', 'IMDB2', 'MR', 'SST-1', 'SST-2', 'SUBJ', 'TREC']

    em_type = 'glove'
    database = 'MR'

    if not os.path.exists(database):
        os.makedirs(database)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 1

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load data
    train_loader, test_loader, num_class = load_dataset(database, em_type)

    for batch_idx, samples in enumerate(train_loader):
        # samples will be a 64 x D dimensional tensor
        # feed it to your neural network model
        print(samples[0].shape)
        print(samples[1].shape)
        break

    A, B, C, D = 64, 8, 16, 16
    # A, B, C, D = 32, 32, 32, 32
    model = capsules(A=A, B=B, C=C, D=D, E=num_class, iters=2).to(device)

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    epochs = 10
    for epoch in range(1, epochs + 1):
        acc = train(train_loader, model, criterion, optimizer, epoch, device)
        acc /= len(train_loader)
        scheduler.step(acc)

    best_acc = test(test_loader, model, criterion, device)
    print('best test accuracy: {:.6f}'.format(best_acc))

    snapshot(model, database, epochs)

if __name__ == '__main__':
    main()

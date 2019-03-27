import time
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from data_helpers import load_dataset
from capsules import *
from spread_loss import *


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred).type(torch.cuda.LongTensor))

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

    epoch_loss = 0
    epoch_acc = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        r = (1. * batch_idx + (epoch - 1) * train_len) / (32 * train_len)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        epoch_loss += loss.item()
        epoch_acc += acc[0].item()

        if batch_idx:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Accuracy: {:.6f}\tLoss: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                acc[0].item(),
                loss.item(),
                batch_time=batch_time,
                data_time=data_time)
            )

    epoch_acc /= train_len
    epoch_loss /= train_len

    return epoch_acc, epoch_loss


def test(test_loader, model, criterion, phase, device):
    model.eval()

    loss = 0
    acc = 0
    test_len = len(test_loader)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()

    loss /= test_len
    acc /= test_len

    print('\n{} set: Average Accuracy: {:.6f}, Loss: {:.6f} \n'.format(phase, acc, loss))
    return acc, loss


def save_plot_to_file(file_name, type, out_name, epochs):
    df = pd.read_csv(file_name)

    df_train = df[df['phase'] == 'train']
    df_dev = df[df['phase'] == 'dev']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[epochs]))

    plt.xlabel('epoch')
    plt.ylabel(type)

    plt.plot(df_train['epoch'], df_train[type], label='train')
    plt.plot(df_dev['epoch'], df_dev[type], label='dev')
    plt.legend()

    plt.savefig(out_name)

def main():
    em_type = 'glove'
    database = 'IMDB'
    folder = database

    if not os.path.exists(database):
        os.makedirs(database)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 1

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load data
    train_loader, dev_loader, test_loader, num_class = load_dataset(database, 64, em_type)

    A, B, C, D = 64, 8, 16, 16
    #A, B, C, D = 32, 32, 32, 32
    model = capsules(A=A, B=B, C=C, D=D, E=num_class, iters=2).to(device)

    # Save the model to the file
    model_file = open(folder + "/model.txt", "w")
    model_file.write('Model:\n{}\n'.format(model))
    model_file.write('Total number of parameters:{}\n'.format(sum(p.numel() for p in model.parameters())))
    model_file.write('Total number of trainable parameters:{}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model_file.close()

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    out_acc = open(folder + "/acc.csv", "w")
    out_loss = open(folder + "/loss.csv", "w")

    out_acc.write('epoch,phase,acc\n')
    out_loss.write('epoch,phase,loss\n')

    epochs = 2
    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()

        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 30)

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, device)

        out_acc.write('{},{},{:.4f}\n'.format(epoch, 'train', train_acc))
        out_loss.write('{},{},{:.4f}\n'.format(epoch, 'train', train_loss))

        dev_acc, dev_loss = test(test_loader, model, criterion, 'dev', device)

        out_acc.write('{},{},{:.4f}\n'.format(epoch, 'dev', dev_acc))
        out_loss.write('{},{},{:.4f}\n'.format(epoch, 'dev', dev_loss))

        scheduler.step(train_acc)

    out_acc.close()
    out_loss.close()
    save_plot_to_file(folder + "/acc.csv", 'acc', folder + "/acc.png", epochs)
    save_plot_to_file(folder + "/loss.csv", 'loss', folder + "/loss.png", epochs)

    test_acc, test_loss = test(test_loader, model, criterion, 'TEST', device)

    out_test = open(folder + "/test.txt", "w")
    out_test.write('Accuracy: {:.6f}, Loss: {:.6f} \n'.format(test_acc, test_loss))
    out_test.close()

    snapshot(model, database, epochs)


if __name__ == '__main__':
    main()
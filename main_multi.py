import os
import torch.optim as optim

from data_helpers import load_dataset
from capsules import *
from spread_loss import *
from main import snapshot, train, test, save_plot_to_file


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 1

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    epochs = 20
    em_types = ['glove', 'word2vec', 'fasttext']
    databases = ["MR", "SST-1", "SST-2", "SUBJ", "TREC", "ProcCons", "IMDB"]
    optimizers = ['adam', 'adagrad']
    schedules = ['ReduceLROnPlateau', 'StepLR']

    save_dir = './multi'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Train
    for em in em_types:
        print('EM {}'.format(em))
        print('*' * 50)

        for d in databases:
            print(d)

            dir = save_dir + '/' + d
            if not os.path.exists(dir):
                os.makedirs(dir)

            # Load data
            train_loader, dev_loader, test_loader, num_class = load_dataset(d, 64, em)

            A, B, C, D = 64, 8, 16, 16
            model = capsules(A=A, B=B, C=C, D=D, E=num_class, iters=2).to(device)

            for o in optimizers:
                for s in schedules:

                    folder = dir + "/em=" + em + ",o=" + o + ",s=" + s
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)

                    if o == 'adam':
                        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
                    elif o == 'adagrad':
                        optimizer = optim.Adagrad(model.parameters(), lr=0.01)

                    if s == 'ReduceLROnPlateau':
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)
                    elif s == 'StepLR':
                        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

                    out_acc = open(folder + "/acc.csv", "w")
                    out_loss = open(folder + "/loss.csv", "w")

                    out_acc.write('epoch,phase,acc\n')
                    out_loss.write('epoch,phase,loss\n')

                    for epoch in range(1, epochs + 1):
                        if s == 'StepLR':
                            scheduler.step()

                        torch.cuda.empty_cache()

                        print('Epoch {}/{}'.format(epoch, epochs))
                        print('-' * 30)

                        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, device)

                        out_acc.write('{},{},{:.4f}\n'.format(epoch, 'train', train_acc))
                        out_loss.write('{},{},{:.4f}\n'.format(epoch, 'train', train_loss))

                        dev_acc, dev_loss = test(dev_loader, model, criterion, 'dev', device)

                        out_acc.write('{},{},{:.4f}\n'.format(epoch, 'dev', dev_acc))
                        out_loss.write('{},{},{:.4f}\n'.format(epoch, 'dev', dev_loss))

                        if s == 'ReduceLROnPlateau':
                            scheduler.step(train_acc)

                    out_acc.close()
                    out_loss.close()
                    save_plot_to_file(folder + "/acc.csv", 'acc', folder + "/acc.png", epochs)
                    save_plot_to_file(folder + "/loss.csv", 'loss', folder + "/loss.png", epochs)

                    test_acc, test_loss = test(test_loader, model, criterion, 'TEST', device)

                    out_test = open(folder + "/test.txt", "w")
                    out_test.write('Accuracy: {:.6f}, Loss: {:.6f} \n'.format(test_acc, test_loss))
                    out_test.close()

                    snapshot(model, folder, epochs)


if __name__ == '__main__':
    main()

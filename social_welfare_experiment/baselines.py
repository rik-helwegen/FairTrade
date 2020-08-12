import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils import load_data

parser = ArgumentParser()
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-hDim', type=int, default=15)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-nIter', type=int, default=501)
parser.add_argument('-batchSize', type=int, default=100)
parser.add_argument('-nTest', type=int, default=9000)
args = parser.parse_args()


class MLP(nn.Module):
    """ PyTorch module, model used for prediction """
    def __init__(self, in_dim=71, h_dim=5, out_dim=1):
        super().__init__()
        # init network
        self.lin1 = nn.Linear(in_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, out_dim)

    def forward(self, x_in):
        h1 = torch.relu(self.lin1(x_in))
        h2 = torch.relu(self.lin2(h1))
        out = torch.sigmoid(self.lin3(h2))

        return out


def get_mlp_acc(x_train, x_test, y_train, y_test):
    # Take number of variables as input dim for model
    n_samples = x_train.shape[0]
    input_dim = x_train.shape[1]
    # Initialise model, new for each rep
    model = MLP(in_dim=input_dim, h_dim=args.hDim, out_dim=y_train.shape[1])
    # Initialise optimiser ADAM
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Maintain loss development for monitoring
    loss_list = []
    test_loss_list = []
    test_loss_idx = []

    n_iter = args.nIter
    batch_size = args.batchSize
    # training loop
    for i in range(n_iter):
        # random index batch
        batch_idx = np.random.choice(a=range(n_samples), size=batch_size, replace=False)
        input = torch.Tensor(x_train[batch_idx])
        # Forward pass through model
        output = model.forward(input)
        # Loss Mean Squared Error
        loss = torch.mean((output - torch.Tensor(y_train[batch_idx])) ** 2)
        # Clear optimizer gradients
        optimizer.zero_grad()
        # Calculate grads
        loss.backward()
        # Optimizer step
        optimizer.step()
        # Append loss to monitor progress
        loss_list.append(loss.detach().numpy())

        # update on progress
        if i % 50 == 0:
            # print('Iteration: %i / %i ' % (i, n_iter))
            test_out = model.forward(torch.Tensor(x_test))
            # Loss Mean Squared Error
            test_loss = torch.mean((test_out - torch.Tensor(y_test)) ** 2)

            test_loss_list.append(test_loss.detach().numpy())
            test_loss_idx.append(i)

    # save loss development
    plt.figure(figsize=(18.0, 12.0))
    plt.plot(loss_list, label='train')
    plt.plot(test_loss_idx, test_loss_list, label='test')
    plt.legend()
    plt.title('Loss development training MLP')
    plt.savefig('baseline_output/loss_develop.png')
    plt.close()

    # test on test set:
    test_out = model.forward(torch.Tensor(x_test))
    test_out = torch.round(test_out)
    test_label = torch.Tensor(y_test)

    correct = torch.sum(test_out == test_label).detach().numpy()
    accuracy_test = correct / len(y_test)

    return accuracy_test


def get_random_forest_acc(x_train, x_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=4)
    rf_classifier.fit(x_train, y_train.ravel())
    y_predict = rf_classifier.predict(x_test)

    correct = np.sum(y_predict[:, np.newaxis] == y_test)
    accuracy = correct/len(y_test)
    return accuracy


def get_logreg_acc(x_train, x_test, y_train, y_test):
    regr = LogisticRegression(solver='liblinear')
    regr.fit(x_train, y_train.ravel())

    y_predict = regr.predict(x_test)
    correct = np.sum(y_predict[:, np.newaxis] == y_test)
    accuracy = correct/len(y_test)
    return accuracy


if __name__ == "__main__":
    acc_list = defaultdict(list)
    print('Obtaining baseline results')

    # repetitions to test stability of training
    for rep_i in range(args.rep):
        # data loader:
        # order of packed data: data = [y, x_con, x_bin, r, b, a]
        train_data, test_data, cat_bin_dict = load_data(n_test=args.nTest)

        # separate label, and concatenate explaining variables
        y_tr = train_data[0]
        y_te = test_data[0]
        x_tr = np.hstack(train_data[1:])
        x_te = np.hstack(test_data[1:])

        if rep_i == 0:
            print('Number of test points: %i' % len(x_te))
        if rep_i == 0:
            print('Number of training points: %i' % len(x_tr))
        acc_list['MLP'].append(get_mlp_acc(x_tr, x_te, y_tr, y_te))
        acc_list['RandomForest'].append(get_random_forest_acc(x_tr, x_te, y_tr, y_te))
        acc_list['LogisticRegression'].append(get_logreg_acc(x_tr, x_te, y_tr, y_te))

    plt.figure(figsize=(18., 12.))
    plt.title('Accuracy Baseline Methods')
    for key, value in acc_list.items():
        plt.xlim(0, 1)
        plt.hist(value, label=key, bins=10, histtype='step')
        plt.legend()
    plt.savefig('baseline_output/accuracy.png')
    plt.close()

    plt.figure(figsize=(18., 12.))
    plt.title('Accuracy Baseline Methods')
    for key, value in acc_list.items():
        np.save('baseline_output/' + key + '.npy', value)
        plt.xlim(0, 1)
        plt.hist(value, label=key, bins=10, alpha=0.6)
        plt.legend()
    plt.xlabel('Accuracy')
    plt.ylabel('Count, total = %i' %args.rep)
    plt.savefig('baseline_output/accuracy_alpha.png')
    plt.close()

from argparse import ArgumentParser
from torch import optim
from collections import defaultdict
from networks import CEVAE_m
from utils import plot_data, train_cevae, plot_loss
import math
import random
import numpy as np

parser = ArgumentParser()
parser.add_argument('-n', type=int, default=125000)
parser.add_argument('-lr', type=float, default=0.002)
parser.add_argument('-dim_h', type=int, default=20)
parser.add_argument('-dim_z', type=int, default=1)
parser.add_argument('-decay', type=int, default=0.2)
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-batchSize', type=int, default=64)
parser.add_argument('-p_a', type=float, default=0.5)
args = parser.parse_args()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def gen_data(args):
    """ ancestral sampling of the dataset """
    print('    - generating data')
    dataset = []

    for i in range(args.n):
        # sample z
        z_i = np.random.normal(0, 1)
        # sample a
        a_i = np.random.binomial(1, args.p_a)
        # set params for x
        mu_x = [-1, 0, 1]
        x_i = []
        # sample each x
        for j in range(len(mu_x)):
            # for x_2, z determines mu
            if j == 1:
                x_i.append(np.random.normal(z_i, max(0.1, 0.55 + 0.2 * z_i)))
            else:
                x_i.append(np.random.normal((1.5 + 1 * a_i) * mu_x[j], max(0.1, 0.55 + 0.2 * z_i)))

        x_i = np.array(x_i)
        beta_a = 3  # 0.1
        beta_x = 2/3  # 1/50
        beta_z = 2  # 1/3
        p_y_i = -8.5 + beta_a * a_i + beta_x * np.sum(np.multiply(x_i, x_i)) + beta_z * z_i
        p_y_i = sigmoid(p_y_i)

        y_i = np.random.binomial(1, p_y_i)

        data_point = [y_i, a_i, z_i]
        data_point = np.concatenate((data_point, x_i))

        dataset.append(data_point)

    dataset = np.array(dataset)

    # train test split
    idx = list(range(dataset.shape[0]))
    random.shuffle(idx)
    len_test = np.round(0.2*len(idx)).astype(int)
    idx_tr = idx[:-len_test]
    idx_te = idx[-len_test:]
    train_data = dataset[idx_tr, :]
    test_data = dataset[idx_te, :]

    return train_data, test_data


if __name__ == "__main__":
    # generate data
    data_tr, data_te = gen_data(args)
    np.save('output/data_tr', data_tr)
    np.save('output/data_te', data_te)

    # plot data
    plot_data(data_tr)

    # initialise cevae network
    CEVAE = CEVAE_m(dim_x=3, dim_z=args.dim_z, dim_h=args.dim_h, dim_a=1)

    # initialise optimizer
    optimizer = optim.Adamax(CEVAE.parameters(), lr=args.lr, weight_decay=args.decay)

    # train cevae network
    loss_dict = defaultdict(list)
    loss_dict = train_cevae(CEVAE, args, optimizer, data_tr, data_te, loss_dict)

    # plot outcomes
    plot_loss(loss_dict)

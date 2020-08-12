"""
# based on;
https://github.com/AMLab-Amsterdam/CEVAE/blob/master/cevae_ihdp.py
"""
from argparse import ArgumentParser

from initialisation import init_qz
from datasets import IHDP
from evaluation import Evaluator, perform_evaluation, plot_figure
from networks import CEVAE_m

import numpy as np
from collections import defaultdict

import torch
from torch import optim

# set random seeds:
# torch.manual_seed(7)
# np.random.seed(7)

parser = ArgumentParser()
# Set Hyperparameters
parser.add_argument('-reps', type=int, default=10)
parser.add_argument('-repTrain', type=int, default=2)
parser.add_argument('-zDim', type=int, default=5)
parser.add_argument('-hDimQ', type=int, default=10)
parser.add_argument('-hDimTar', type=int, default=10)
parser.add_argument('-epochs', type=int, default=400)
parser.add_argument('-batch', type=int, default=50)
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-decay', type=float, default=0.)
parser.add_argument('-printEvery', type=int, default=1)
parser.add_argument('-nSamplesZ', type=int, default=2)
parser.add_argument('-comment', type=str, default='None')
# tStart: number of epochs only training t dist by weighting loss
parser.add_argument('-tStart', type=int, default=80)
args = parser.parse_args()

dataset = IHDP(replications=args.reps)
# number of continuous features, see datasets/columns file
x_con_n = 5
x_bin_n = 19
x_dim = [x_bin_n, x_con_n]

# Loop for replications DGP
for rep_i, (train, valid, test) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication %i/%i' % (rep_i + 1, args.reps))
    # loop for replications CEVAE training on same data
    for train_i in range(args.repTrain):

        # read out data
        (xtr, atr, ttr, ytr), (y_cftr, mutr, ztr, t_cftr) = train
        (xva, ava, tva, yva), (y_cfva, muva, zva, t_cfva) = valid
        (xte, ate, tte, yte), (y_cfte, mute, zte, t_cfte) = test

        if (x_con_n + x_bin_n) != xtr.shape[1]:
            raise ValueError('Check dimensionality x')

        # reorder features with binary first and continuous after
        perm = [b for b in range(x_con_n, xtr.shape[1])] + [c for c in range(x_con_n)]
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

        # set evaluator objects
        evaluator_train = Evaluator(y_cftr, atr, t_cftr, mutr)
        evaluator_test = Evaluator(y_cfte, ate, t_cfte, mute)

        x_con_mean = np.mean(xtr[:, x_bin_n:], 0)
        x_con_sd = np.std(xtr[:, x_bin_n:], 0)
        xtr[:, x_bin_n:] = (xtr[:, x_bin_n:] - x_con_mean)/x_con_sd
        # normalize test set with train metrics, like in new data case
        xte[:, x_bin_n:] = (xte[:, x_bin_n:] - x_con_mean)/x_con_sd

        CEVAE = CEVAE_m(z_dim=args.zDim, h_dim_tar=args.hDimTar, h_dim_q=args.hDimQ, x_bin_n=x_bin_n, x_con_n=x_con_n)
        optimizer = optim.Adamax(CEVAE.parameters(), lr=args.lr, weight_decay=args.decay)

        # set batch size
        M = args.batch
        n_epoch, n_iter_per_epoch, idx = args.epochs, 4 * int(xtr.shape[0] / M), list(range(xtr.shape[0]))

        # init dictionaries to save progress values
        loss, rmse, pehe, t_std = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        loss_test, rmse_test, pehe_test, t_std_test = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

        for epoch in range(n_epoch):

            for j in range(n_iter_per_epoch):

                # get indices for random batch
                batch = np.random.choice(idx, M, replace=False)
                # select data for batch
                x_train = torch.cuda.FloatTensor(xtr[batch])
                a_train = torch.cuda.FloatTensor(atr[batch])
                t_train = torch.cuda.FloatTensor(ttr[batch])
                y_train = torch.cuda.FloatTensor(ytr[batch])

                data_batch = [x_train, a_train, t_train, y_train]

                loss, objective, _, _ = CEVAE.forward(data_batch, loss, args, epoch, x_dim)

                optimizer.zero_grad()
                # Calculate gradients
                objective.backward()
                # Update step
                optimizer.step()

            # Evaluation
            if epoch % args.printEvery == 0:
                # obtain loss for test set:
                a_test, x_test, y_test, t_test = torch.cuda.FloatTensor(ate), torch.cuda.FloatTensor(xte), \
                                                 torch.cuda.FloatTensor(yte), torch.cuda.FloatTensor(tte)
                filename = ''
                for key, value in vars(args).items():
                    if key == 'reps':
                        key = 'rep'
                        value = rep_i
                    if key == 'repTrain':
                        value = train_i
                    filename += str(key) + '-' + str(value) + '_'

                # Save model
                torch.save(CEVAE.state_dict(), 'models/' + filename + '.pt')

                print('Epoch %i' % epoch)
                print('Results on training set:')
                pehe, rmse, t_std = perform_evaluation(CEVAE, torch.tensor(xtr).cuda(), torch.tensor(ttr).cuda(),
                                                       torch.tensor(atr).cuda(), evaluator_train, pehe, rmse, t_std,
                                                       filename)
                print('Results on test set:')
                pehe_test, rmse_test, t_std_test, loss_test = perform_evaluation(CEVAE, x_test, t_test, a_test,
                                                                                 evaluator_test, pehe_test, rmse_test,
                                                                                 t_std, filename, loss_test, x_bin_n,
                                                                                 x_con_n, y_test)
                print('___________________')

                plot_figure(loss, rmse, t_std, pehe, args.tStart, args.nSamplesZ, n_iter_per_epoch,
                            'TRAIN_' + filename.replace('.', 'dot') + '.png')
                plot_figure(loss_test, rmse_test, t_std_test, pehe_test, args.tStart, args.nSamplesZ, 1,
                            'TEST_' + filename.replace('.', 'dot'))  # test is evaluated 1 time per epoch



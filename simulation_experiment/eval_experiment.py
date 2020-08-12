from collections import defaultdict
from networks import CEVAE_m
import numpy as np
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import statsmodels.discrete.discrete_model as sm
from sklearn.ensemble import RandomForestClassifier

parser = ArgumentParser()
parser.add_argument('-n', type=int, default=62500)
parser.add_argument('-dim_h', type=int, default=20)
parser.add_argument('-dim_z', type=int, default=1)
parser.add_argument('-n_rep', type=int, default=20)
args = parser.parse_args()


def compare_recon(x, data_recon, reconstruct=False):
    idx_a0 = np.where(data_recon[:, 1] == 0)
    idx_a1 = np.where(data_recon[:, 1] == 1)
    # Plot overview x distribution
    plt.figure(figsize=(5, 4), dpi=110)
    num_x = int(data_recon.shape[1] - 3)
    for i in range(num_x):
        plt.subplot(num_x, 1, i + 1)
        if i == (num_x - 1):
            plt.xlabel('x value')
        plt.ylabel('Hist x' + str(i + 1))
        plt.xlim(-5, 5)

        label0, label1 = 'a0', 'a1'
        plt.hist(x[idx_a0, i][0], alpha=0.2, label=label0, bins=50, color='firebrick')
        plt.hist(x[idx_a1, i][0], alpha=0.2, label=label1, bins=50, color='navy')

        label0 = 'recon a0'
        label1 = 'recon a1'
        plt.hist(data_recon[idx_a0, 3 + i][0], histtype='step', label=label0, bins=50, color='firebrick')
        plt.hist(data_recon[idx_a1, 3 + i][0], histtype='step', label=label1, bins=50, color='navy')

        plt.legend()
    plt.tight_layout()
    plt.savefig('output/data_x_recon')

    plt.show()


if __name__ == "__main__":

    loss_dict = defaultdict(list)

    # initialise cevae network
    CEVAE = CEVAE_m(dim_x=3, dim_z=args.dim_z, dim_h=args.dim_h, dim_a=1)
    CEVAE.load_state_dict(torch.load('output/cevae_model'))

    # load data
    data_tr = np.load('output/data_tr.npy')
    data_te = np.load('output/data_te.npy')

    # unpack train
    y_tr = data_tr[:, 0]
    a_tr = data_tr[:, 1]
    x_tr = data_tr[:, 3:]

    # unpack test
    y_te = data_te[:, 0]
    a_te = data_te[:, 1]
    x_te = data_te[:, 3:]
    # create counterfactual sensitive var
    a_te_cf = 1 - a_te

    outcomes = defaultdict(list)

    # repetition loop to check stability
    for i in range(args.n_rep):

        # create reconstruction data for test
        x_te_recon, y_te_recon, z_te_recon = CEVAE.forward(torch.Tensor(x_te), torch.Tensor(a_te), torch.Tensor(y_te),
                                                           loss_dict, reconstruct=True)

        # generate *_test_reconstruction_counterfactual
        x_te_recon_cf, y_te_recon_cf, z_te_recon_cf = CEVAE.forward(torch.Tensor(x_te), torch.Tensor(a_te_cf),
                                                                    torch.Tensor(y_te), loss_dict, reconstruct=True)

        def torch_sample_np(x):
            return x.sample().numpy()


        # group reconstruction data
        recon = [x_te_recon, y_te_recon, z_te_recon]
        # group counterfactual reconstruction data
        recon_cf = [x_te_recon_cf, y_te_recon_cf, z_te_recon_cf]

        # sample and numpy
        x_te_recon, y_te_recon, z_te_recon = [torch_sample_np(data) for data in [x_te_recon, y_te_recon, z_te_recon]]
        x_te_recon_cf, y_te_recon_cf, z_te_recon_cf = [torch_sample_np(data) for data in
                                                       [x_te_recon_cf, y_te_recon_cf, z_te_recon_cf]]

        # show graph for to evaluate reconstruction output
        # compare_recon(x_te, np.array(np.concatenate([y_te_recon_cf, a_te[:, np.newaxis], z_te_recon_cf, x_te_recon], 1)))

        # Fit models using train setRF_ypred_te
        input_tr = np.concatenate((np.ones(len(a_tr))[:, np.newaxis], x_tr, a_tr[:, np.newaxis]), 1)
        # logistic regression
        LR = sm.Logit(y_tr, input_tr)
        LR_fit = LR.fit()

        RF = RandomForestClassifier(n_estimators=10, max_depth=4)
        RF.fit(input_tr, y_tr)

        # ------------------ test model accuracy on test set
        input_te = np.concatenate((np.ones(len(a_te))[:, np.newaxis], x_te, a_te[:, np.newaxis]), 1)

        def get_accuracies(input):
            # normal LR
            LR_ypred_te = LR_fit.predict(input)
            # normal LR with fixed a
            input_te_adjusted = input.copy()
            input_te_adjusted[:, -1] = 0
            LR_adj_ypred_te = LR_fit.predict(input_te_adjusted)
            # random forest
            RF_ypred_te = RF.predict(input)

            def accuracy(y_hat, y_true):
                return np.sum(np.round(y_hat) == y_true) / len(y_true)

            LR_acc = accuracy(LR_ypred_te, y_te)
            LR_adj_acc = accuracy(LR_adj_ypred_te, y_te)
            RF_acc = accuracy(RF_ypred_te, y_te)

            return LR_acc, LR_adj_acc, RF_acc

        LR_acc, LR_adj_acc, RF_acc = get_accuracies(input_te)
        outcomes['LR_acc_te'].append(LR_acc)
        outcomes['LR_adj_acc_te'].append(LR_adj_acc)
        outcomes['RF_acc'].append(RF_acc)

        # ------------------ test model accuracy on reconstructed test set

        input_te_recon = np.concatenate((np.ones(len(a_te))[:, np.newaxis], x_te_recon, a_te[:, np.newaxis]), 1)
        LR_acc, LR_adj_acc, RF_acc = get_accuracies(input_te_recon)
        outcomes['LR_acc_te_recon'].append(LR_acc)
        outcomes['LR_adj_acc_te_recon'].append(LR_adj_acc)
        outcomes['RF_acc_te_recon'].append(RF_acc)

        # ------------------ test model accuracy on reconstructed test set
        input_te_recon_cf = np.concatenate((np.ones(len(a_te))[:, np.newaxis], x_te_recon_cf, a_te_cf[:, np.newaxis]), 1)
        LR_acc, LR_adj_acc, RF_acc = get_accuracies(input_te_recon_cf)
        outcomes['LR_acc_te_recon_cf'].append(LR_acc)
        outcomes['LR_adj_acc_te_recon_cf'].append(LR_adj_acc)
        outcomes['RF_acc_te_recon_cf'].append(RF_acc)

        # ------------------ test counterfactual fairness scores (cfs)
        outcomes['LR_cfs'].append(np.mean(np.abs(LR_fit.predict(input_te_recon) - LR_fit.predict(input_te_recon_cf))))

        # for adjusted LR, we adjust input
        input_te_recon_adj = input_te_recon.copy()
        input_te_recon_adj[:, -1] = 0
        input_te_recon_cf_adj = input_te_recon_cf.copy()
        input_te_recon_cf_adj[:, -1] = 0

        outcomes['LR_adj_cfs'].append(np.mean(np.abs(LR_fit.predict(input_te_recon_adj) - LR_fit.predict(input_te_recon_cf_adj))))

        outcomes['RF_cfs'].append(np.mean(np.abs(RF.predict(input_te_recon) - RF.predict(input_te_recon_cf))))

    for key, values in outcomes.items():
        print(key + ': ' + str(np.mean(values)) + ' (' + str(np.std(values)) + ')')

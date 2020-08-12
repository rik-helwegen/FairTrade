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
args = parser.parse_args()


def compare_recon(x, data_recon, reconstruct=False):

    idx_a0 = np.where(data_recon[:, 1] == 0)
    idx_a1 = np.where(data_recon[:, 1] == 1)
    # Plot overview x distribution
    plt.figure(figsize=(5, 4), dpi=110)
    num_x = int(data_recon.shape[1]-3)
    for i in range(num_x):
        plt.subplot(num_x, 1, i + 1)
        if i == (num_x-1):
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
    y = torch.Tensor(data_te[:, 0])
    a = torch.Tensor(data_te[:, 1])
    x = torch.Tensor(data_te[:, 3:])

    # create reconstruction data
    x_recon, y_recon, z_recon = CEVAE.forward(x, a, y, loss_dict, reconstruct=True)
    # create counterfactual reconstruction data, a not used in q(z|x), so forward suffices
    a_cf = 1-a
    x_recon_cf, y_recon_cf, z_recon_cf = CEVAE.forward(x, a_cf, y, loss_dict, reconstruct=True)

    # from torch to numpy
    x_recon = x_recon.sample().numpy()
    x_recon_cf = x_recon_cf.sample().numpy()
    y_recon = y_recon.sample().numpy()
    y_recon_cf = y_recon_cf.sample().numpy()
    z_recon = z_recon.sample().numpy()
    z_recon_cf = z_recon_cf.sample().numpy()
    a = a.numpy()[:, np.newaxis]
    y = y.numpy()
    a_cf = a_cf.numpy()[:, np.newaxis]

    data_recon = np.array(np.concatenate((y_recon, a, z_recon, x_recon), 1))

    X_recon = np.concatenate((np.ones(len(a))[:, np.newaxis], x_recon, a), 1)
    X_cf = np.concatenate((np.ones(len(a))[:, np.newaxis], x_recon_cf, a_cf), 1)

    compare_recon(x, data_recon, reconstruct=True)

    X = np.concatenate((np.ones(len(a))[:, np.newaxis], x.numpy(), a), 1)

    # fit with train data
    y_tr = torch.Tensor(data_tr[:, 0])
    a_tr = torch.Tensor(data_tr[:, 1])
    x_tr = torch.Tensor(data_tr[:, 3:])
    a_tr = a_tr.numpy()[:, np.newaxis]
    X_tr = np.concatenate((np.ones(len(a_tr))[:, np.newaxis], x_tr.numpy(), a_tr), 1)
    y_tr = y_tr.numpy()

    LR_1 = sm.Logit(y_tr, X_tr)
    LR1_fit = LR_1.fit()

    LR1_ypred = LR1_fit.predict(X)
    LR1_ypred_Xrecon = LR1_fit.predict(X_recon)

    LR1_ypred_cf = LR1_fit.predict(X_cf)
    LR1_counterfactual_fairness_error = np.mean(np.abs(LR1_ypred_cf - LR1_ypred_Xrecon))
    print('LR1 counterfactual fairness error;')
    print(LR1_counterfactual_fairness_error)

    # adjusted Logit, fix input A to baseline value
    def fair_LR_predic(LR, X):
        X_out = X.copy()
        X_out[:, -1] = 0
        return LR.predict(X_out)

    LR1_ypred_recon_fair = fair_LR_predic(LR1_fit, X_recon)
    LR1_ypred_cf_fair = fair_LR_predic(LR1_fit, X_cf)
    LR1_counterfactual_fairness_error_fair = np.mean(np.abs(LR1_ypred_recon_fair - LR1_ypred_cf_fair))
    print('LR1 -adjusted- counterfactual fairness error;')
    print(LR1_counterfactual_fairness_error_fair)

    RF_cf_error_list = []
    for i in range(10):
        RF = RandomForestClassifier(n_estimators=10, max_depth=4)
        RF_fair = RandomForestClassifier(n_estimators=10, max_depth=4)
        RF.fit(X_tr, y_tr)

        RF_ypred = RF.predict(X)
        RF_ypred_Xrecon = RF.predict(X_recon)

        RF_ypred_Xrecon = RF.predict(X_recon)
        RF_ypred_Xcf = RF.predict(X_cf)
        RF_counterfactual_fairness_error = np.mean(np.abs(RF_ypred_Xrecon - RF_ypred_Xcf))

        RF_cf_error_list.append(RF_counterfactual_fairness_error)

    print('RF counterfactual fairness error;')
    print('mean')
    print(np.mean(RF_cf_error_list))
    print('sd')
    print(np.std(RF_cf_error_list))

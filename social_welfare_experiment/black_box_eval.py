import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils import load_data, CEVAE_mod, scatter_latent

parser = ArgumentParser()
parser.add_argument('-nTest', type=int, default=9000)
parser.add_argument('-hDim', type=int, default=50)
parser.add_argument('-zDim', type=int, default=5)
parser.add_argument('-rep', type=int, default=0)
parser.add_argument('-genSize', type=int, default=5000)
parser.add_argument('-evalIter', type=int, default=500)
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-comment', type=str, default='')
args = parser.parse_args()


def generate_counterfactual():
    train_data, test_data, cat_bin_dict = load_data(n_test=args.nTest)
    y_tr, x_tr_con, x_tr_bin, r_tr, b_tr, a_tr = train_data
    x_tr = np.hstack((x_tr_bin, x_tr_con))

    CEVAE = CEVAE_mod(args=args, dim_x=x_tr.shape[1], dim_b=b_tr.shape[1], dim_a=1, dim_x_con=x_tr_con.shape[1],
                      dim_x_bin=cat_bin_dict, dim_z=args.zDim, dim_r=r_tr.shape[1], dim_q_h=args.hDim,
                      dim_p_h=args.hDim).to(args.device)

    load_model = './model_path.pt'
    CEVAE.load_state_dict(torch.load(load_model))

    # select a batch from the data
    batch_idx = np.random.choice(a=range(x_tr.shape[0]), size=args.genSize, replace=False)
    batch_data = [torch.Tensor(g[batch_idx]).to(args.device) for g in train_data]

    loss_dict = defaultdict(list)
    reconstruction = CEVAE.forward(batch_data, args, loss_dict, cat_bin_dict, reconstruct=True)
    reconstruction_cf = CEVAE.forward(batch_data, args, loss_dict, cat_bin_dict, reconstruct=True, switch_a=True)

    x_tr_recon = vec_from_recon(batch_data, reconstruction, cat_bin_dict)
    x_tr_recon_cf = vec_from_recon(batch_data, reconstruction_cf, cat_bin_dict)

    np.save('counterfact_data/x_tr_recon.npy', x_tr_recon)
    np.save('counterfact_data/y_tr.npy', batch_data[0])
    np.save('counterfact_data/x_tr_recon_cf.npy', x_tr_recon_cf)


def vec_from_recon(original_data, reconstruction, cat_bin_dict):
    """ reconstruct vector in same index order as original, categorical information stored in cat_bin_dict """
    # obtain original vector for shape
    x_batch_bin = original_data[2]
    # create a placeholder batch of vectors for the reconstructions
    x_batch_bin_recon = np.zeros(x_batch_bin.shape)
    # select the (not correctly ordered) reconstructions
    x_bin_dict = reconstruction['x_bin_dict']
    # loop through the different categorical distributions
    x_bin_categories = list(cat_bin_dict.keys())
    for x_bin_category in x_bin_categories:
        if x_bin_category == 'BINARY':
            # for the binary case all elements represent a distribution
            round_vec = torch.round(x_bin_dict[x_bin_category].probs)
            x_batch_bin_recon[:, cat_bin_dict[x_bin_category]] = round_vec.detach()
        else:
            # for categorical cases, we sample a reconstruction
            dist = x_bin_dict[x_bin_category]
            dist_sample = dist.sample()
            x_batch_bin_recon[:, cat_bin_dict[x_bin_category]] = dist_sample

    a = original_data[5]
    b = original_data[4]

    x_con = reconstruction['x_con']
    r = reconstruction['r']

    # do not round continuous, but do round mean
    vector = np.hstack([x_con.sample().detach(), x_batch_bin_recon, r.sample().detach(), b, a])

    return vector


def get_logreg_acc(x_train, x_test, y_train, y_test):
    regr = LogisticRegression(solver='liblinear')
    regr.fit(x_train, y_train.ravel())

    y_predict = regr.predict(x_test)

    correct = np.sum(y_predict[:, np.newaxis] == y_test)
    accuracy = correct/len(y_test)
    return accuracy


if __name__ == "__main__":
    for rep_i in range(args.rep):
        generate_counterfactual()

        # data loader:
        train_data, test_data, cat_bin_dict = load_data(n_test=args.nTest)

        # separate label, and concatenate explaining variables
        y_tr = train_data[0]
        y_te = test_data[0]
        x_tr = np.hstack(train_data[1:])
        x_te = np.hstack(test_data[1:])

        regr = LogisticRegression(solver='liblinear')
        regr.fit(x_tr, y_tr.ravel())
        y_predict = regr.predict(x_te)
        print('accuracy original data LR; ' + str(np.sum(y_predict[:, np.newaxis] == y_te)/ len(y_te)))

        RF = RandomForestClassifier(n_estimators=10, max_depth=4)
        RF.fit(x_tr, y_tr.ravel())
        RF_ypred = RF.predict(x_te)
        print('accuracy original data RF; ' + str(np.sum(np.round(RF_ypred) == y_te) / len(y_te)))

        # -- predict for reconstruction data;
        x_tr_recon = np.load('counterfact_data/x_tr_recon.npy')
        y_tr = np.load('counterfact_data/y_tr.npy')

        y_predict_x_recon = regr.predict(x_tr_recon)
        correct = np.sum(np.equal(y_predict_x_recon[:, np.newaxis], y_tr))
        accuracy = correct / len(y_tr)
        print('accuracy reconstruction data LR; ' + str(accuracy))

        RF_ypred_recon = RF.predict(x_tr_recon)
        print('accuracy reconstruction data RF; ' + str(np.sum(np.round(RF_ypred_recon[:, np.newaxis]) == y_tr) / len(y_tr)))




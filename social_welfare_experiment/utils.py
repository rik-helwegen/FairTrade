import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
from torch.distributions import OneHotCategorical


def load_data(n_test=500):
    """ Load prepared files, return train test set """
    print('loading preprocessed data sets.')
    # load cleaned and balanced data
    path = './'
    label_y = np.load(path + 'clean_balanced/label_y.npy')
    base_b = np.load(path + 'clean_balanced/base_b.npy')
    covariates_x_bin = np.load(path + 'clean_balanced/covariates_x_bin.npy')
    covariates_x_con = np.load(path + 'clean_balanced/covariates_x_con.npy')
    ethnicity_a = np.load(path + 'clean_balanced/sensitive_a.npy')
    resolving_r = np.load(path + 'clean_balanced/resolving_r.npy', allow_pickle=True)
    column_names = np.load('clean_balanced/columns_dict.npy', allow_pickle='TRUE').item()

    # group categorical variables within binary values
    cat_bin_dict = defaultdict(list)
    for i, column in enumerate(column_names['covariates_x_bin']):
        # for categorical dummies; first three symbols overlap
        cat_bin_dict[column[:3]].append(i)
    cat_bins = list(cat_bin_dict.keys())
    for key in cat_bins:
        # Group binary (non-categorical) variables as one recognizable 'category'
        if len(cat_bin_dict[key]) == 1:
            cat_bin_dict['BINARY'].append(cat_bin_dict[key][0])
            # the variably is now indexed by 'BINARY', so original should be removed
            del cat_bin_dict[key]

    # def dependent and independent vars
    y = label_y.astype(float)
    # select sensitive a: etnicity
    a = ethnicity_a.astype(int)
    # select base b: gender and age dummies, remove ethnicity
    b = base_b.astype(float)
    # resolving as float
    r = resolving_r.astype(float)
    x_bin = covariates_x_bin.astype(int)
    x_con = covariates_x_con.astype(float)

    data = [y, x_con, x_bin, r, b, a]

    print('number of total datapoints: %i' % len(y))

    # random train/test split
    idx_list = list(range(len(y)))
    np.random.shuffle(idx_list)
    idx_train = idx_list[:-n_test]
    idx_test = idx_list[-n_test:]
    data_train = [g[idx_train] for g in data]
    data_test = [g[idx_test] for g in data]

    return data_train, data_test, cat_bin_dict


class q_z_mod(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.mu = nn.Linear(dim_h, dim_out)
        self.sigma = nn.Linear(dim_h, dim_out)
        self.softplus = nn.Softplus()

    def forward(self, observations):
        h1 = F.elu(self.lin1(observations))
        mu = self.mu(h1)
        sigma = self.softplus(self.sigma(h1))
        out = normal.Normal(mu, sigma)

        return out


class p_x_zba_mod(nn.Module):
    def __init__(self, dim_in, dim_h, con_out_dim, bin_out_dict):
        """ bin_out_dim is a dictionary giving the output dimensions per category"""
        super().__init__()
        # save required vars
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2 = nn.Linear(dim_h, dim_h)

        # TAR structure
        self.mu_a0 = nn.Linear(dim_h, con_out_dim)
        self.mu_a1 = nn.Linear(dim_h, con_out_dim)

        # TAR structure
        self.sigma_a0 = nn.Linear(dim_h, con_out_dim)
        self.sigma_a1 = nn.Linear(dim_h, con_out_dim)

        self.softplus = nn.Softplus()
        # initialise a separate output layer for each category, in which all binary variables are included as 1 category
        self.headnames = list(bin_out_dict.keys())
        # need to stay in same order as headnames, to be recognised correctly in forward
        # TAR structure, for each output distribution a head for a=0 and a=1
        self.binheads_a0 = nn.ModuleList(nn.Linear(dim_h, len(i)) for i in bin_out_dict.values())
        self.binheads_a1 = nn.ModuleList(nn.Linear(dim_h, len(i)) for i in bin_out_dict.values())

    def forward(self, zb, a):
        h1 = F.elu(self.lin1(zb))
        h2 = F.elu(self.lin2(h1))

        # finish forward binary and categorical covariates
        bin_out_dict = dict()

        # for each categorical variable
        for i in range(len(self.headnames)):
            # calculate probability paramater
            p_a0 = self.binheads_a0[i](h2)
            p_a1 = self.binheads_a1[i](h2)
            dist_p_a0 = torch.sigmoid(p_a0)
            dist_p_a1 = torch.sigmoid(p_a1)
            # create distribution in dict
            if self.headnames[i] == 'BINARY':
                bin_out_dict[self.headnames[i]] = bernoulli.Bernoulli((1-a)*dist_p_a0 + a*dist_p_a1)
            else:
                bin_out_dict[self.headnames[i]] = OneHotCategorical((1-a)*dist_p_a0 + a*dist_p_a1)

        # finish forward continuous vars for the right TAR head
        mu_a0 = self.mu_a0(h2)
        mu_a1 = self.mu_a1(h2)
        sigma_a0 = self.softplus(self.sigma_a0(h2))
        sigma_a1 = self.softplus(self.sigma_a1(h2))
        # cap sigma to prevent collapse for continuous vars being 0
        sigma_a0 = torch.clamp(sigma_a0, min=0.1)
        sigma_a1 = torch.clamp(sigma_a1, min=0.1)
        con_out = normal.Normal((1-a) * mu_a0 + a * mu_a1, (1-a)* sigma_a0 + a * sigma_a1)

        return con_out, bin_out_dict


class p_y_zbaxr_mod(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)

    def forward(self, zb, a):
        h1 = F.elu(self.lin1(zb))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out


class p_r_zbax_mod(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_h)
        self.lin2_a0 = nn.Linear(dim_h, dim_out)
        self.lin2_a1 = nn.Linear(dim_h, dim_out)

    def forward(self, zb, a):
        h1 = F.elu(self.lin1(zb))
        h2_a0 = F.elu(self.lin2_a0(h1))
        h2_a1 = F.elu(self.lin2_a1(h1))
        bern_p_a0 = torch.sigmoid(h2_a0)
        bern_p_a1 = torch.sigmoid(h2_a1)

        bern_out = bernoulli.Bernoulli((1-a)*bern_p_a0 + a*bern_p_a1)

        return bern_out


class CEVAE_mod(nn.Module):
    def __init__(self, args, dim_x, dim_b, dim_a, dim_x_con, dim_x_bin, dim_z=5, dim_r=0, dim_q_h=100, dim_p_h=50):
        super().__init__()
        # Change structure in case of resolving variables
        self.q_z = q_z_mod(dim_in=dim_x + dim_b + dim_a + dim_r, dim_h=dim_q_h, dim_out=dim_z)
        self.p_r_zbax = p_r_zbax_mod(dim_in=dim_z + dim_b + dim_x, dim_h=dim_p_h, dim_out=dim_r)
        self.p_x_zba = p_x_zba_mod(dim_in=dim_z + dim_b, dim_h=dim_p_h, con_out_dim=dim_x_con, bin_out_dict=dim_x_bin)
        self.p_y_zbaxr = p_y_zbaxr_mod(dim_in=dim_z + dim_b + dim_r + dim_x, dim_h=dim_p_h, dim_out=1)
        self.p_z_dist = normal.Normal(torch.zeros(args.zDim).to(args.device), torch.ones(args.zDim).to(args.device))

    def forward(self, batch_data, args, loss_dict, cat_bin_dict, eval=False, reconstruct=False, switch_a=False):
        y_batch, x_batch_con, x_batch_bin, r_batch, b_batch, a_batch = batch_data
        # INFER distribution over z
        z_infer = self.q_z.forward(observations=torch.cat((x_batch_con, x_batch_bin, r_batch, b_batch, a_batch), 1))

        # in case multiple samples are used to approximate lower bound
        sample_losses = torch.tensor([]).to(args.device)
        for _ in range(args.nSamplesZ):
            # use mean of z distribution for evaluation
            if eval:
                z_infer_sample = z_infer.mean
            else:
                z_infer_sample = z_infer.rsample()

            # optional INTERVENTION of switching all a values
            if switch_a:
                a_batch = 1 - a_batch

            # RECONSTRUCTION LOSS
            # loss for covariates
            x_con, x_bin_dict = self.p_x_zba.forward(torch.cat((z_infer_sample, b_batch), 1), a_batch)
            l1 = x_con.log_prob(x_batch_con).sum(1)
            loss_dict['Reconstr_x_con'].append(l1.mean().cpu().detach().float())
            # Separate loss value per category, and one for the group of bernouilli distributions
            # create zero vector to sum loss, make of the right shape
            l2 = torch.zeros(y_batch.shape[0])
            x_bin_categories = list(cat_bin_dict.keys())
            for x_bin_category in x_bin_categories:
                # select correct estimated distribution
                dist = x_bin_dict[x_bin_category]
                # select the observed value, using the indices in the dict
                obs = x_batch_bin[:, cat_bin_dict[x_bin_category]]

                # add log probability
                # the 'binary' category consist of multiple distributions, so we need an extra sum to get a logp per obs
                if x_bin_category == 'BINARY':
                    l2 += dist.log_prob(obs).sum(1)
                else:
                    l2 += dist.log_prob(obs)

            loss_dict['Reconstr_x_bin'].append(l2.mean().cpu().detach().float())

            # loss for resolving variables
            r = self.p_r_zbax.forward(torch.cat((z_infer_sample, b_batch, x_batch_bin, x_batch_con), 1), a_batch)
            l5 = r.log_prob(r_batch).sum(1)
            loss_dict['Reconstr_r'].append(l5.mean().cpu().detach().float())

            # loss for label recovery
            y_infer = self.p_y_zbaxr.forward(torch.cat((z_infer_sample, b_batch,  x_batch_bin, x_batch_con, r_batch), 1), a_batch)
            l3 = y_infer.log_prob(y_batch).squeeze()
            loss_dict['Reconstr_y'].append(l3.mean().cpu().detach().float())

            # REGULARIZATION LOSS
            l4 = (self.p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
            loss_dict['Regularization'].append(l4.mean().cpu().detach().float())

            # collect loss for this z sample, take mean over batch
            loss = -torch.mean(l1 + l2 + l3 + l4 + l5)
            # append loss for visualisation / tracking of loss
            loss_dict['Total'].append(loss.cpu().detach().numpy())
            # concatenate loss for this z-sample, to be used in backward pass
            sample_losses = torch.cat((sample_losses, loss.unsqueeze(0)))

            if eval:
                y_out = torch.round(y_infer.mean)
                accuracy = torch.sum(y_out == y_batch).cpu().detach().numpy() / x_batch_con.shape[0]
                return accuracy, loss, z_infer_sample

            # when reconstructing, return reconstructed distributions
            if reconstruct:
                reconstruction = {
                    'x_con': x_con,
                    'x_bin_dict': x_bin_dict,
                    'r': r,
                    'y': y_infer
                }
                return reconstruction

        # take mean over objective for z samples
        objective = torch.mean(sample_losses)

        return objective, loss_dict


def scatter_latent(z, condition, progress):
    rep_i, i = progress
    # plot scatter z, for insight independence Z, A
    z_tsne = TSNE(n_components=2).fit_transform(z.cpu().detach().numpy())
    plt.figure(figsize=(4, 4))
    plt.plot(z_tsne[np.where(condition.cpu().detach().numpy() == 0)[0], 0],
                z_tsne[np.where(condition.cpu().detach().numpy() == 0)[0], 1],
                'o', label='a=0', color='red', mfc='none')
    plt.plot(z_tsne[np.where(condition.cpu().detach().numpy() == 1)[0], 0],
                z_tsne[np.where(condition.cpu().detach().numpy() == 1)[0], 1],
                '+', label='a=1', color='blue')
    plt.legend()
    plt.savefig('output/scatter_z_rep' + str(rep_i) + '_iter' + str(i) + '.png')
    plt.tight_layout()
    plt.close()


def plot_loss():
    dir = './output/output/'
    train_list = []
    test_list = []
    y_train_list = []
    y_test_list = []
    for file in os.listdir(dir):
        if 'components' not in file:
            if file[:5] == 'train':
                train_loss = np.load(dir + file)
                train_list.append(train_loss)
            if file[:4] == 'test':
                test_loss = np.load(dir + file)
                test_list.append(test_loss)
            if file[:7] == 'y_train':
                y_train_list.append(np.load(dir+file))
            if file[:6] == 'y_test':
                y_test_list.append(np.load(dir+file))

    train_array = np.array(train_list)
    test_array = np.array(test_list)

    x_train = range(train_array.shape[1])
    x_test = np.linspace(0, train_array.shape[1], test_array.shape[1])

    train_mean = np.mean(train_array, axis=0)
    test_mean = np.mean(test_array, axis=0)
    train_std = np.std(train_array, axis=0)
    test_std = np.std(test_array, axis=0)

    plt.title('Loss development CEVAE (+/- std of 20 repetitions)')
    plt.plot(x_train, train_mean, color='red', label='Mean train loss (+/- std)')
    plt.plot(x_test, test_mean, color='blue', label='Mean test loss (+/- std)')
    plt.fill_between(x_train, train_mean + train_std, train_mean - train_std, color='red', alpha=0.4)
    plt.fill_between(x_test, test_mean + test_std, test_mean - test_std, color='blue', alpha=0.4)
    plt.xlabel('Training iteration')
    plt.ylabel('Loss value')
    plt.legend()
    plt.show()

    plt.title('Reconstruction loss y CEVAE (+/- std of 20 repetitions) ')
    y_train_mean = np.mean(np.array(y_train_list), axis=0)
    y_test_mean = np.mean(np.array(y_test_list), axis=0)
    plt.plot(y_test_mean)
    plt.show()
    y_train_std = np.std(np.array(y_train_list), axis=0)
    y_test_std = np.std(np.array(y_test_list), axis=0)
    plt.plot(x_train, y_train_mean, color='red', label='Mean train reconstruction loss y (+/- std)')
    plt.plot(x_test, y_test_mean, color='blue', label='Mean test reconstruction loss y (+/- std)')
    plt.fill_between(x_train, y_train_mean + y_train_std, y_train_mean - y_train_std, color='red', alpha=0.4)
    plt.fill_between(x_test, y_test_mean + y_test_std, y_test_mean - y_test_std, color='blue', alpha=0.4)
    plt.xlabel('Training iteration')
    plt.ylabel('Loss value')
    plt.legend()
    plt.show()


def plot_cpe_curve():
    print("printing cpe curve")
    accuracy_dict = np.load('./output/cpe_output/accuracy_dict.npy').item()
    statpar_a0 = np.load('./output/cpe_output/stat_par_y1a0_dict.npy').item()
    statpar_a1 = np.load('./output/cpe_output/stat_par_y1a1_dict.npy').item()
    dict_keys = list(statpar_a0.keys())
    print(dict_keys)
    observ_len = len(statpar_a0[dict_keys[0]])

    # get differences of statistical parity parts
    statpar_error = defaultdict(list)
    colors = ['orange', 'green', 'red', 'blue', 'brown']
    ax = plt.subplot(111)
    # plt.title('Causal Path Enabler Curve')
    color_count = 0
    for key in dict_keys:
        x = accuracy_dict[key]
        for i in range(observ_len):
            statpar_error[key].append(np.absolute(statpar_a1[key][i] - statpar_a0[key][i]))
        y = statpar_error[key]

        # for visualisation turn around y-axis
        y = np.ones(len(y))-y

        # scatter points on background
        ax.scatter(x, y, color=colors[color_count], alpha=0.2)
        color_count += 1
    ax.legend()

    color_count = 0
    # Write plot values to excel
    with pd.ExcelWriter('./output/pse_numerical.xlsx') as writer:
        # loop over different specifications of the model
        for key in dict_keys:
            # create label
            label = 'avg aux('
            for i in key:
                label += (i + ', ')
            label = (label[:-2] + ')')

            x = np.mean(accuracy_dict[key])
            y = np.mean(statpar_error[key])

            df_acc = pd.DataFrame({'Accuracy': accuracy_dict[key]})
            df_stat = pd.DataFrame({'Statistical Parity': statpar_error[key]})
            df_acc.to_excel(writer, sheet_name=key, startcol=0, index=False, header='Accuracy')
            df_stat.to_excel(writer, sheet_name=key, startcol=1, index=False, header='Statistical parity')

            # for visualisation turn around y-axis, see def. stat par score in paper
            y = 1-y
            ax.scatter(x, y, label=label, color=colors[color_count], s=100)
            color_count += 1
    ax.legend()
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Statistical Parity')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

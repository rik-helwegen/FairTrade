from networks import CEVAE_m, CFM_m
from datasets import IHDP
from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

parser = ArgumentParser()
parser.add_argument('-reps', type=int, default=10)
parser.add_argument('-hDimF', type=int, default=100)
parser.add_argument('-nIter', type=int, default=800)
parameters = parser.parse_args()

x_con_n = 5
x_bin_n = 19
x_dim = [x_bin_n, x_con_n]


def train_cfm(target, z_dist, a_train=None, t_train=None):
    """
    Training a Counter Factual fair prediction Model
    - If A and T are given as arguments in the call, they are used as input for the CFM
    """

    # dimensionality CFM
    input_dim = z_dist.sample().shape[1]
    if a_train is not None:
        input_dim += 1
    if t_train is not None:
        input_dim += 1
    output_dim = target.shape[1]

    print('-> training CFM with input dim: %i' % input_dim)

    CFM = CFM_m(input_dim, parameters.hDimF, output_dim).cuda()
    # increase init noise, otherwise model incidentally doesn't learn
    optimizer = torch.optim.RMSprop(CFM.parameters(), lr=0.0005)

    loss_cfm_list = []
    # training loop
    for i in range(parameters.nIter):
        # do not use .rsample() on z_dist, we don't want to update CEVAE
        input = z_dist.sample()
        if a_train is not None:
            input = torch.cat((input, torch.cuda.FloatTensor(a_train)), 1)
        if t_train is not None:
            input = torch.cat((input, torch.cuda.FloatTensor(t_train)), 1)

        y_est = CFM(input)

        loss_cfm = torch.mean((y_est - target) ** 2)
        optimizer.zero_grad()
        loss_cfm.backward()
        optimizer.step()
        loss_cfm_list.append(loss_cfm.cpu().detach().numpy())

    return CFM


def conditional_results(results, condition):
    """ Separate outcome distribution into conditional np outcome distributions"""

    results_0 = results[condition == 0].cpu().detach().numpy()
    results_1 = results[condition == 1].cpu().detach().numpy()

    return [results_0, results_1]


def pse_results(cevae_file, data, data_results):
    """Gather results for the different models of interest"""

    # read data
    (train, valid, test) = data
    (xtr, atr, ttr, ytr), (y_cftr, mutr, ztr, t_cftr) = train
    (xva, ava, tva, yva), (y_cfva, muva, zva, t_cfva) = valid
    (xte, ate, tte, yte), (y_cfte, mute, zte, t_cfte) = test

    # normalisation, based on train data stat, like in real deployment
    x_con_mean = np.mean(xtr[:, x_bin_n:], 0)
    x_con_sd = np.std(xtr[:, x_bin_n:], 0)
    xtr[:, x_bin_n:] = (xtr[:, x_bin_n:] - x_con_mean) / x_con_sd
    xva[:, x_bin_n:] = (xva[:, x_bin_n:] - x_con_mean) / x_con_sd
    xte[:, x_bin_n:] = (xte[:, x_bin_n:] - x_con_mean) / x_con_sd

    data_train = [torch.cuda.FloatTensor(d) for d in train[0]]
    data_test = [torch.cuda.FloatTensor(d) for d in test[0]]
    y_train = torch.cuda.FloatTensor(ytr)

    # Init and load real world model
    CEVAE = CEVAE_m(z_dim=parameters.zDim, h_dim_tar=parameters.hDimTar,
                    h_dim_q=parameters.hDimQ, x_bin_n=x_bin_n, x_con_n=x_con_n)
    CEVAE.load_state_dict(torch.load(cevae_file))

    # ------ True Y results ------ #
    data_results['True Y'].append([yte[ate == 0], yte[ate == 1]])

    # ------ CEVAE Y results ------ #
    loss = defaultdict(list)  # loss not used, but required for cevae
    # data as torch tensor
    _, _, z_infer_test, y_mean_test = CEVAE.forward(data_test, loss, parameters, parameters.tStart + 1, x_dim)
    data_results['CEVAE Y'].append(conditional_results(y_mean_test, torch.cuda.FloatTensor(ate)))

    # ------ CFM Y(T,A,Z) results ------ #
    _, _, z_infer_train, y_mean_train = CEVAE.forward(data_train, loss, parameters, parameters.tStart + 1, x_dim)
    CFM = train_cfm(y_train, z_infer_train, a_train=atr, t_train=ttr)
    input = torch.cat((z_infer_test.mean, torch.cuda.FloatTensor(ate), torch.cuda.FloatTensor(tte)), 1)
    data_results['CPE Y(Z,A,T)'].append(conditional_results(CFM(input), torch.cuda.FloatTensor(ate)))

    # ------ CFM Y(T,Z) results ------ #
    CFM = train_cfm(y_train, z_infer_train, t_train=ttr)
    input = torch.cat((z_infer_test.mean, torch.cuda.FloatTensor(tte)), 1)
    data_results['CPE Y(Z,T)'].append(conditional_results(CFM(input), torch.cuda.FloatTensor(ate)))

    # ------ CFM Y(Z) results ------ #
    CFM = train_cfm(y_train, z_infer_train)
    data_results['CPE Y(Z)'].append(conditional_results(CFM(z_infer_test.mean), torch.cuda.FloatTensor(ate)))

    return data_results


def vis_results(data_results):
    """Create set of histograms with outcomes to compare models"""

    plt.figure(figsize=(18.0, 12.0))
    subidx = 1
    binedges = np.linspace(0, 28, 50)
    bincenters = 0.5 * (np.array(binedges[1:]) + np.array(binedges[:-1])) - 0.1

    for key, values in data_results.items():
        plt.subplot(5, 1, subidx)
        # stack outcomes over different repetitions
        stack_a0 = []
        stack_a1 = []
        # formulate outcomes in bin counts for histogram
        for est_out in values:
            stack_entry_a0, _ = np.histogram(est_out[0], bins=binedges)
            stack_entry_a1, _ = np.histogram(est_out[1], bins=binedges)
            stack_a0.append(stack_entry_a0)
            stack_a1.append(stack_entry_a1)
        stack_a0 = np.vstack(stack_a0)
        stack_a1 = np.vstack(stack_a1)

        # mean and average over bin counts:
        mean_a0, std_a0 = np.mean(stack_a0, 0), np.std(stack_a0, 0)
        mean_a1, std_a1 = np.mean(stack_a1, 0), np.std(stack_a1, 0)

        # Exclude non converged variances, no interpretation and very high
        std_a0[0] = 0
        std_a1[0] = 0

        plt.bar(bincenters - 0.15, mean_a0, width=0.3, color='orange', yerr=std_a0, ecolor='orange', alpha=0.6, label='a=0')
        plt.bar(bincenters + 0.15, mean_a1, width=0.3, color='b', yerr=std_a1, ecolor='b', alpha=0.6, label='a=1')
        plt.title(key)
        plt.xlabel('Y values')
        plt.ylabel('Prediction value count')
        plt.legend()
        subidx += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.2)
    plt.savefig('results/compareY.png')


if __name__ == "__main__":
    directory = './models/test_models/'
    dataset = IHDP(replications=parameters.reps)
    # store result data
    data_results = defaultdict(list)
    # loop through different data generation instances (replications)
    for rep_i, data in enumerate(dataset.get_train_valid_test()):
        print('Data repetition %i' % rep_i)

        # loop through different files of trained CEVAEs
        for cevae_file in os.listdir(directory):

            # only select models trained on this dataset, standard case just 1
            if cevae_file.endswith('.pt') and int(cevae_file[4]) == rep_i:

                # Recover arguments from filename
                for model_par in cevae_file.split('/')[-1][:-4].split('_'):
                    # split into key and value, dtype depending on variable
                    key, value = model_par.split('-')
                    if key in ['lr', 'decay']:
                        setattr(parameters, key, float(value))
                    elif key == 'comment':
                        setattr(parameters, key, str(value))
                    else:
                        setattr(parameters, key, int(value))

                # create models and save outcomes
                data_results = pse_results(directory + cevae_file, data, data_results)

    # Visualise results
    np.save('pse_cfm_data.npy', data_results)
    vis_results(data_results)


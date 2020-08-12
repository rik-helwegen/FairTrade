import numpy as np
import matplotlib.pyplot as plt
import torch


def eval_cevae(CEVAE, data_te, loss_dict):
    y_te = torch.Tensor(data_te[:, 0])
    a_te = torch.Tensor(data_te[:, 1])
    x_te = torch.Tensor(data_te[:, 3:])

    _, loss_dict = CEVAE.forward(x_te, a_te, y_te, loss_dict, evaluate=True)

    torch.save(CEVAE.state_dict(), 'output/cevae_model')

    return loss_dict


def train_cevae(CEVAE, args, optimizer, data_tr, data_te, loss_dict):
    print('    - training CEVAE')
    # Training loop
    y = torch.Tensor(data_tr[:, 0])
    a = torch.Tensor(data_tr[:, 1])
    x = torch.Tensor(data_tr[:, 3:])

    loss_dict = eval_cevae(CEVAE, data_te, loss_dict)

    for epoch_i in range(args.epochs):
        iter_per_epoch = np.floor(y.shape[0]/args.batchSize).astype(int)

        for iter_i in range(iter_per_epoch):

            batch_idx = np.random.choice(list(range(x.shape[0])), args.batchSize, replace=False)

            # select data for batch
            x_batch = torch.Tensor(x[batch_idx])
            a_batch = torch.Tensor(a[batch_idx])
            y_batch = torch.Tensor(y[batch_idx])

            neg_lowerbound, loss_dict = CEVAE.forward(x_batch, a_batch, y_batch, loss_dict)

            optimizer.zero_grad()
            # Calculate gradients
            neg_lowerbound.backward()
            # Update step
            optimizer.step()

        loss_dict = eval_cevae(CEVAE, data_te, loss_dict)

    return loss_dict


def plot_data(data, reconstruct=False):

    idx_a0 = np.where(data[:, 1] == 0)
    idx_a1 = np.where(data[:, 1] == 1)
    # Plot overview x distribution
    plt.figure(figsize=(5, 4), dpi=100)
    # plt.suptitle('Distribution x, conditioned on a')
    # loop through 5 x variables
    num_x = int(data.shape[1]-3)
    for i in range(num_x):
        plt.subplot(num_x, 1, i + 1)
        if i == (num_x-1):
            plt.xlabel('x value')
        plt.ylabel('Hist x' + str(i + 1))
        plt.xlim(-5, 5)

        label0, label1 = 'a0', 'a1'
        if reconstruct:
            label0 = 'recon a0'
            label1 = 'recon a1'
        plt.hist(data[idx_a0, 3 + i][0], histtype='step', label=label0, bins=50, color='firebrick')
        plt.hist(data[idx_a1, 3 + i][0], histtype='step', label=label1, bins=50, color='navy')

        plt.legend()
    plt.tight_layout()
    if reconstruct:
        plt.savefig('output/data_x_recon')
    else:
        plt.savefig('output/data_x')
    plt.show()

    # plot y distributions
    plt.figure(figsize=(5, 4), dpi=100)
    # plt.suptitle('Distribution Y, conditioned on a')
    plt.subplot(2, 1, 1)
    plt.hist(data[idx_a0, 0][0], label='a0', histtype='step', alpha=0.6, color='firebrick')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.hist(data[idx_a1, 0][0], label='a1', histtype='step', alpha=0.6, color='navy')
    plt.legend()
    if reconstruct:
        plt.savefig('output/data_y_recon')
    else:
        plt.savefig('output/data_y')
    plt.show()
    plt.close()


def plot_loss(loss_dict):
    plt.figure(figsize=(16, 8), dpi=100)
    subplot = 1
    # divide by 2, because every kind has a training an evaluation entry which should be plotted as one
    max_subplot = len(loss_dict)/2
    for key, value in loss_dict.items():

        if key[:5] != 'eval_':
            plt.subplot(max_subplot, 1, subplot)
            # calculate how many training points there are per evaluation point
            train_test_obs_ratio = np.floor(len(value)/len(loss_dict['eval_' + key]))
            test_iter = [train_test_obs_ratio * i for i in range(len(loss_dict['eval_' + key]))]

            plt.plot(value, color='blue', label='train')
            plt.plot(test_iter, loss_dict['eval_' + key], color='red', label='test')
            plt.xlabel('iteration')
            plt.ylabel('log probability')
            plt.legend()
            plt.title(key)

            subplot += 1

    plt.tight_layout()
    plt.savefig('output/loss')
    # plt.show()

from collections import defaultdict
from networks import CEVAE_m
import numpy as np
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

parser = ArgumentParser()
parser.add_argument('-n', type=int, default=62500)
parser.add_argument('-dim_h', type=int, default=20)
parser.add_argument('-dim_z', type=int, default=1)
args = parser.parse_args()


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

    # create counterfactual reconstruction data
    x_1, _, _ = CEVAE.forward(x, torch.ones(len(a)), y, loss_dict, reconstruct=True)
    x_0, _, _ = CEVAE.forward(x, torch.zeros(len(a)), y, loss_dict, reconstruct=True)

    x_recon = x_recon.sample().numpy()
    x_0 = x_0.sample().numpy()
    x_1 = x_1.sample().numpy()

    a = a.numpy()[:, np.newaxis]

    idx_a0 = np.where(a == 0)
    idx_a1 = np.where(a == 1)

    n_bins = 22

    plt.figure(figsize=(5, 4), dpi=110)
    plt.subplot(3,1,1)
    plt.xlim(-5, 5)
    plt.hist(x[idx_a0, 0][0], alpha=0.2, bins=n_bins, label='a=0', color='firebrick')
    plt.hist(x[idx_a1, 0][0], alpha=0.2, bins=n_bins, label='a=1', color='navy')
    plt.hist(x_recon[idx_a0,0][0], bins=20, histtype='step', label='recon a=0', color='firebrick')
    plt.hist(x_recon[idx_a1,0][0], bins=20, histtype='step', label='recon a=1', color='navy')
    plt.ylabel('Hist x1')
    plt.legend()
    plt.subplot(3,1,2)
    plt.xlim(-5, 5)
    plt.hist(x[idx_a0, 0][0], alpha=0.2, bins=n_bins, label='a=0', color='firebrick')
    plt.hist(x[idx_a1, 0][0], alpha=0.2, bins=n_bins, label='a=1', color='navy')
    plt.hist(x_0[idx_a0,0][0], bins=20, histtype='step', label='recon a=0, do(a=1)', color='firebrick')
    plt.hist(x_0[idx_a1,0][0], bins=20, histtype='step', label='recon a=1', color='navy')
    plt.ylabel('Hist x1')
    plt.legend()
    plt.subplot(3,1,3)
    plt.xlim(-5, 5)
    plt.hist(x[idx_a0, 0][0], alpha=0.2, bins=n_bins, label='a=0', color='firebrick')
    plt.hist(x[idx_a1, 0][0], alpha=0.2, bins=n_bins, label='a=1', color='navy')
    plt.hist(x_1[idx_a0,0][0], bins=20, histtype='step', label='recon a=0', color='firebrick')
    plt.hist(x_1[idx_a1,0][0], bins=20, histtype='step', label='recon a=1, do(a=0)', color='navy')
    plt.xlabel('Value x1')
    plt.ylabel('Hist x1')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/data_int.png')
    plt.show()

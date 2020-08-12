import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import CEVAE_mod, load_data

parser = ArgumentParser()
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-hDim', type=int, default=50)
parser.add_argument('-nTest', type=int, default=1000)
parser.add_argument('-nTrain', type=int, default=1000)
parser.add_argument('-zDim', type=int, default=5)
parser.add_argument('-rep', type=int, default=20)
parser.add_argument('-nIter', type=int, default=10001)
parser.add_argument('-batchSize', type=int, default=512)
parser.add_argument('-nSamplesZ', type=int, default=1)
parser.add_argument('-evalIter', type=int, default=500)
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-comment', type=str, default='')
args = parser.parse_args()


# Model class for auxiliary model
class aux_m(nn.Module):
    def __init__(self, input_dim=20, h_dim_f=100, out_dim=1):
        super().__init__()
        # init network
        self.lin1 = nn.Linear(input_dim, h_dim_f)
        self.lin2 = nn.Linear(h_dim_f, out_dim)

    def forward(self, z_in):
        h1 = F.elu(self.lin1(z_in))
        out = torch.sigmoid(self.lin2(h1))
        return out


if __name__ == "__main__":

    acc_list = defaultdict(list)
    stat_par_y1a0 = defaultdict(list)
    stat_par_y1a1 = defaultdict(list)

    # repeat whole training process to observe stability
    for rep_i in range(args.rep):
        print('rep: %i' % rep_i)
        # data loader:
        # order of packed data: data = [y, x_con, x_bin, r, b, a]
        train_data, test_data, cat_bin_dict = load_data(n_test=args.nTest)

        # unpack train x in order to init in right shapes
        train_data = [data[:args.nTrain] for data in train_data]
        y_tr, x_tr_con, x_tr_bin, r_tr, b_tr, a_tr = train_data
        x_tr = np.hstack((x_tr_bin, x_tr_con))

        CEVAE = CEVAE_mod(args=args, dim_x=x_tr.shape[1], dim_b=b_tr.shape[1], dim_a=1, dim_x_con=x_tr_con.shape[1],
                          dim_x_bin=cat_bin_dict, dim_z=args.zDim, dim_r=r_tr.shape[1], dim_q_h=args.hDim,
                          dim_p_h=args.hDim).to(args.device)

        model = './model_path.pt'
        CEVAE.load_state_dict(torch.load(model))

        # init Causal Path Enabler (auxiliary -fair- models)
        path_combinations = ['z', 'zb', 'zbr', 'zbrx', 'zbrxa']
        AUX = dict()
        AUX['z'] = aux_m(input_dim=args.zDim, h_dim_f=100)
        AUX['zb'] = aux_m(input_dim=args.zDim + b_tr.shape[1], h_dim_f=100)
        AUX['zbr'] = aux_m(input_dim=args.zDim + b_tr.shape[1] + r_tr.shape[1], h_dim_f=100)
        AUX['zbrx'] = aux_m(input_dim=args.zDim + b_tr.shape[1] + r_tr.shape[1] + x_tr.shape[1], h_dim_f=100)
        AUX['zbrxa'] = aux_m(input_dim=args.zDim + b_tr.shape[1] + r_tr.shape[1] + x_tr.shape[1] + a_tr.shape[1],
                             h_dim_f=100)

        # init optimizer
        optimizer = dict()
        for combination in path_combinations:
            optimizer[combination] = torch.optim.RMSprop(AUX[combination].parameters(), lr=args.lr)

        # Maintain loss development for monitoring
        loss_dict = defaultdict(list)

        # training loop
        for i in range(args.nIter):
            # select random batch
            batch_idx = np.random.choice(a=range(x_tr.shape[0]), size=args.batchSize, replace=False)
            batch_data = [torch.Tensor(g[batch_idx]).to(args.device) for g in train_data]

            y_batch, x_batch_con, x_batch_bin, r_batch, b_batch, a_batch = batch_data
            # INFER distribution over z, using inference network of CEVAE
            z_infer = CEVAE.q_z.forward(
                observations=torch.cat((x_batch_con, x_batch_bin, r_batch, b_batch, a_batch), 1))
            # No need to store derivative to CEVAE
            z_sample = z_infer.sample().detach()

            aux_output = dict()
            aux_output['z'] = AUX['z'].forward(z_sample)
            aux_output['zb'] = AUX['zb'].forward(torch.cat((z_sample, b_batch), 1))
            # use R(x,Z,B,A) below instead in order to exclude A->X->R->Y
            aux_output['zbr'] = AUX['zbr'].forward(torch.cat((z_sample, b_batch, r_batch), 1))
            aux_output['zbrx'] = AUX['zbrx'].forward(
                torch.cat((z_sample, b_batch, r_batch, x_batch_con, x_batch_bin), 1))
            aux_output['zbrxa'] = AUX['zbrxa'].forward(
                torch.cat((z_sample, b_batch, r_batch, x_batch_con, x_batch_bin, a_batch), 1))

            # calculate loss and update step
            objective = dict()
            for combination in path_combinations:
                objective[combination] = torch.mean((aux_output[combination] - y_batch) ** 2)
                optimizer[combination].zero_grad()
                objective[combination].backward()
                optimizer[combination].step()

                loss_dict[combination].append(float(objective[combination].cpu().detach().numpy()))

        # test on test set
        batch_data = [torch.Tensor(g).to(args.device) for g in test_data]

        y_batch, x_batch_con, x_batch_bin, r_batch, b_batch, a_batch = batch_data
        # INFER distribution over z, using inference network of CEVAE
        z_infer = CEVAE.q_z.forward(observations=torch.cat((x_batch_con, x_batch_bin, r_batch, b_batch, a_batch), 1))
        # No need to store derivative to CEVAE
        z_sample = z_infer.sample().detach()

        aux_output = dict()
        aux_output['z'] = AUX['z'].forward(z_sample)
        aux_output['zb'] = AUX['zb'].forward(torch.cat((z_sample, b_batch), 1))
        aux_output['zbr'] = AUX['zbr'].forward(torch.cat((z_sample, b_batch, r_batch), 1))
        aux_output['zbrx'] = AUX['zbrx'].forward(torch.cat((z_sample, b_batch, r_batch, x_batch_con, x_batch_bin), 1))
        aux_output['zbrxa'] = AUX['zbrxa'].forward(
            torch.cat((z_sample, b_batch, r_batch, x_batch_con, x_batch_bin, a_batch), 1))

        # ------ Test Statistical parity -----
        mask_a0 = (a_batch == 0).squeeze()
        mask_a1 = (a_batch == 1).squeeze()

        for combination in path_combinations:
            y_predict = torch.round(aux_output[combination])
            accuracy = torch.sum(y_predict == y_batch).cpu().detach().numpy() / y_batch.shape[0]
            acc_list[combination].append(accuracy)
            p_y1_a0 = y_predict[mask_a0].sum() / mask_a0.sum()
            p_y1_a1 = y_predict[mask_a1].sum() / mask_a1.sum()
            stat_par_y1a0[combination].append(p_y1_a0.cpu().detach().numpy())
            stat_par_y1a1[combination].append(p_y1_a1.cpu().detach().numpy())

    np.save('accuracy_dict.npy', acc_list)
    np.save('stat_par_y1a0_dict.npy', stat_par_y1a0)
    np.save('stat_par_y1a1_dict.npy', stat_par_y1a1)

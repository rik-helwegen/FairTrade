import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import bernoulli, normal


class q_z_x_m(nn.Module):
    def __init__(self, dim_in=5, dim_h=20, dim_out=1):
        super().__init__()

        self.input = nn.Linear(dim_in, dim_h)
        self.h1 = nn.Linear(dim_h, dim_h)
        self.h2 = nn.Linear(dim_h, dim_h)
        self.mu_z = nn.Linear(dim_h, dim_out)
        self.sigma_z = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        # No need to include a because y and a are independent

        x_embed = F.elu(self.input(x))
        h1 = F.elu(self.h1(x_embed))
        h2 = F.elu(self.h2(h1))
        z = normal.Normal(self.mu_z(h2), torch.exp(self.sigma_z(h2)))
        return z


class p_x_za_m(nn.Module):
    def __init__(self, dim_in=2, dim_h=10, dim_out=5):
        super().__init__()

        self.input = nn.Linear(dim_in, dim_h)
        self.h1 = nn.Linear(dim_h, dim_h)
        self.h2 = nn.Linear(dim_h, dim_h)
        self.mu_x = nn.Linear(dim_h, dim_out)
        self.sigma_x = nn.Linear(dim_h, dim_out)

    def forward(self, za):
        za_embed = F.elu(self.input(za))
        # No need for TAR heads because of simplicity and small dimensionality
        h1 = F.elu(self.h1(za_embed))
        h2 = F.elu(self.h2(h1))
        x = normal.Normal(self.mu_x(h2), torch.exp(self.sigma_x(h2)))
        return x


class p_y_xza_m(nn.Module):
    def __init__(self, dim_in=6, dim_h=20, dim_out=1):
        super().__init__()

        self.input = nn.Linear(dim_in, dim_h)
        self.h1 = nn.Linear(dim_h, dim_h)
        self.h2_a0 = nn.Linear(dim_h, dim_h)
        self.h2_a1 = nn.Linear(dim_h, dim_h)
        self.h3_a0 = nn.Linear(dim_h, dim_h)
        self.h3_a1 = nn.Linear(dim_h, dim_h)
        self.p_y_a0 = nn.Linear(dim_h, dim_out)
        self.p_y_a1 = nn.Linear(dim_h, dim_out)

    def forward(self, xz, a):
        xz_embed = F.elu(self.input(xz))
        h1 = F.elu(self.h1(xz_embed))
        # Separate TAR heads for a values
        h2_a0 = F.elu(self.h2_a0(h1))
        h2_a1 = F.elu(self.h2_a1(h1))
        h3_a0 = F.elu(self.h3_a0(h2_a0))
        h3_a1 = F.elu(self.h3_a1(h2_a1))
        p_y_a0 = torch.sigmoid(self.p_y_a0(h3_a0))
        p_y_a1 = torch.sigmoid(self.p_y_a1(h3_a1))
        y = bernoulli.Bernoulli((1-a)*p_y_a0 + a*p_y_a1)
        return y


class CEVAE_m(nn.Module):
    def __init__(self, dim_x, dim_z, dim_h, dim_a):
        super().__init__()
        self.q_z_x = q_z_x_m(dim_in=dim_x, dim_h=dim_h, dim_out=1)
        self.p_x_za = p_x_za_m(dim_in=dim_z + dim_a, dim_h=dim_h, dim_out=dim_x)
        self.p_y_xza = p_y_xza_m(dim_in=dim_x + dim_z, dim_h=dim_h, dim_out=1)
        self.standard_normal = normal.Normal(torch.zeros(dim_z), torch.ones(dim_z))

    def forward(self, x, a, y, loss_dict, evaluate=False, reconstruct=False):
        z_infer = self.q_z_x(x)
        z_sample = z_infer.rsample()
        # use single z sample to approximate lowerbound
        cat_za = torch.cat([z_sample, a.unsqueeze(1)], 1)
        x_recon = self.p_x_za(cat_za)
        cat_xz = torch.cat([x, z_sample], 1)
        y_recon = self.p_y_xza(cat_xz, a.unsqueeze(1))

        # loss calculation (z bern so no regularisation needed)
        # - reconstruction
        # sum over x dimensions
        l1 = x_recon.log_prob(x).sum(1).unsqueeze(1)

        l2 = y_recon.log_prob(y.unsqueeze(1))

        # - regularization
        l3 = (self.standard_normal.log_prob(z_sample) - z_infer.log_prob(z_sample))

        if reconstruct:
            return x_recon, y_recon, z_infer

        # we want to maximise the lowerbound, so minimise the negative lowerbound
        # neg_lowerbound = -torch.mean(l1 + l2 + l3)
        neg_lowerbound = -torch.mean(l1 + l2)

        # store values for progress analysis
        if evaluate:
            loss_dict['eval_reconstruction x'].append(l1.mean().detach().float())
            loss_dict['eval_reconstruction y'].append(l2.mean().detach().float())
            loss_dict['eval_regularization'].append(l3.mean().detach().float())
            loss_dict['eval_Negative lowerbound'].append(neg_lowerbound.detach().float())
        else:
            loss_dict['reconstruction x'].append(l1.mean().detach().float())
            loss_dict['reconstruction y'].append(l2.mean().detach().float())
            loss_dict['regularization'].append(l3.mean().detach().float())
            loss_dict['Negative lowerbound'].append(neg_lowerbound.detach().float())

        return neg_lowerbound, loss_dict

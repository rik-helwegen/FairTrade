import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import bernoulli, normal

# class naming convention: p_A_BC -> p(A|B,C)


class p_x_za(nn.Module):

    def __init__(self, dim_in=11, nh=3, dim_h=20, dim_out_bin=17, dim_out_con=6):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out_bin = dim_out_bin
        self.dim_out_con = dim_out_con

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh-1)])
        # output layer defined separate for continuous and binary outputs
        self.output_bin = nn.Linear(dim_h, dim_out_bin)
        # for each output an mu and sigma are estimated
        self.output_con_mu = nn.Linear(dim_h, dim_out_con)
        self.output_con_sigma = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, za_input):
        z = F.elu(self.input(za_input))
        for i in range(self.nh-1):
            z = F.elu(self.hidden[i](z))
        # for binary outputs:
        x_bin_p = torch.sigmoid(self.output_bin(z))
        x_bin = bernoulli.Bernoulli(x_bin_p)
        # for continuous outputs
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        # sigma overruled by simplicity assumption Madras, legitimized by standardization
        sigma = torch.exp(2*torch.ones(mu.shape).cuda())
        x_con = normal.Normal(mu, sigma)

        if (z != z).all():
            print('Forward contains NaN')

        return x_bin, x_con


class p_t_za(nn.Module):

    def __init__(self, dim_in=10, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # no variable amount of layers for clarity
        # dim_in is dim of latent space z
        self.rep = nn.Linear(dim_in, dim_h)
        self.lin0 = nn.Linear(dim_h, dim_h)
        self.output_a0 = nn.Linear(dim_h, dim_out)
        self.lin1 = nn.Linear(dim_h, dim_h)
        self.output_a1 = nn.Linear(dim_h, dim_out)

    def forward(self, z, a):
        rep = F.elu(self.rep(z))
        # for each value of a different mapping from representation
        rep0 = F.elu(self.lin0(rep))
        rep1 = F.elu(self.lin1(rep))
        p_a0 = torch.sigmoid(self.output_a0(rep0))
        p_a1 = torch.sigmoid(self.output_a1(rep1))
        # combine TAR net into single output
        out = bernoulli.Bernoulli((1-a)*p_a0 + a*p_a1)
        return out


class p_y_zta(nn.Module):

    def __init__(self, dim_in=10, dim_h=100, dim_rep=20):
        super().__init__()

        # Nested TARnet for a, t values
        self.h = nn.Linear(dim_in, dim_h)
        self.rep = nn.Linear(dim_h, dim_rep)
        self.h_a0 = nn.Linear(dim_rep, dim_h)
        self.h_a1 = nn.Linear(dim_rep, dim_h)
        self.rep_a0 = nn.Linear(dim_h, dim_rep)
        self.rep_a1 = nn.Linear(dim_h, dim_rep)
        self.h_a0_t0 = nn.Linear(dim_rep, dim_h)
        self.h_a0_t1 = nn.Linear(dim_rep, dim_h)
        self.h_a1_t0 = nn.Linear(dim_rep, dim_h)
        self.h_a1_t1 = nn.Linear(dim_rep, dim_h)
        self.mu_a0_t0 = nn.Linear(dim_h, 1)
        self.mu_a0_t1 = nn.Linear(dim_h, 1)
        self.mu_a1_t0 = nn.Linear(dim_h, 1)
        self.mu_a1_t1 = nn.Linear(dim_h, 1)
        self.sigma_a0_t0 = nn.Linear(dim_h, 1)
        self.sigma_a0_t1 = nn.Linear(dim_h, 1)
        self.sigma_a1_t0 = nn.Linear(dim_h, 1)
        self.sigma_a1_t1 = nn.Linear(dim_h, 1)

    def forward(self, z, t, a):
        # Separated forwards for different t values, TAR

        h = F.elu(self.h(z))
        rep = F.elu(self.rep(h))
        h_a0 = F.elu(self.h_a0(rep))
        h_a1 = F.elu(self.h_a1(rep))
        rep_a0 = F.elu(self.rep_a0(h_a0))
        rep_a1 = F.elu(self.rep_a1(h_a1))
        h_a0_t0 = F.elu(self.h_a0_t0(rep_a0))
        h_a0_t1 = F.elu(self.h_a0_t1(rep_a0))
        h_a1_t0 = F.elu(self.h_a1_t1(rep_a1))
        h_a1_t1 = F.elu(self.h_a1_t1(rep_a1))
        mu_a0_t0 = F.elu(self.mu_a0_t0(h_a0_t0))
        mu_a0_t1 = F.elu(self.mu_a0_t1(h_a0_t1))
        mu_a1_t0 = F.elu(self.mu_a1_t0(h_a1_t0))
        mu_a1_t1 = F.elu(self.mu_a1_t1(h_a1_t1))
        sigma_a0_t0 = torch.exp(self.sigma_a0_t0(h_a0_t0))
        sigma_a0_t1 = torch.exp(self.sigma_a0_t1(h_a0_t1))
        sigma_a1_t0 = torch.exp(self.sigma_a1_t0(h_a1_t0))
        sigma_a1_t1 = torch.exp(self.sigma_a1_t1(h_a1_t1))

        mu = (1-a)*(1-t) * mu_a0_t0 + \
             (1-a) * t * mu_a0_t1 + \
             a * (1-t) * mu_a1_t0 + \
             a * t * mu_a1_t1
        sigma = (1 - a) * (1 - t) * sigma_a0_t0 + \
             (1 - a) * t * sigma_a0_t1 + \
             a * (1 - t) * sigma_a1_t0 + \
             a * t * sigma_a1_t1

        # set mu according to t value
        y = normal.Normal(mu, sigma)

        return y


class q_z_xa(nn.Module):

    def __init__(self, dim_in=25, nh=1, dim_h=20, dim_out=10):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh

        # Shared layers with separated output layers

        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])

        self.mu = nn.Linear(dim_h, dim_out)
        self.sigma = nn.Linear(dim_h, dim_out)

    def forward(self, xa):

        x = F.elu(self.input(xa))

        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))

        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))

        z = normal.Normal(mu, sigma)
        return z


class CEVAE_m(nn.Module):
    def __init__(self, z_dim=20, h_dim_tar=100, h_dim_q=10, x_bin_n=19, x_con_n=5):
        super().__init__()

        # init networks (overwritten per replication)
        self.p_x_za_dist = p_x_za(dim_in=z_dim + 1, nh=1, dim_h=20, dim_out_bin=x_bin_n,
                                  dim_out_con=x_con_n).cuda()
        self.p_t_za_dist = p_t_za(dim_in=z_dim, nh=1, dim_h=h_dim_tar, dim_out=1).cuda()
        self.p_y_zta_dist = p_y_zta(dim_in=z_dim, dim_h=h_dim_tar, dim_rep=20).cuda()
        self.q_z_xa_dist = q_z_xa(dim_in=x_bin_n + x_con_n + 1, nh=1,
                                  dim_h=h_dim_q, dim_out=z_dim).cuda()
        self.p_z_dist = normal.Normal(torch.zeros(z_dim).cuda(), torch.ones(z_dim).cuda())

    def forward(self, data_batch, loss, args, epoch, x_dim, interv_a=None):
        x_train, a_train, t_train, y_train = data_batch
        x_bin_n, x_con_n = x_dim

        # inferred distribution over z
        xa = torch.cat((x_train, a_train), 1)
        z_infer = self.q_z_xa_dist(xa=xa)

        sample_losses = torch.tensor([]).cuda()
        sample_y_mean = []

        # use 'n_samples_z' samples to approximate expectation in lowerbound VAE
        for i in range(args.nSamplesZ):
            z_infer_sample = z_infer.rsample()
            # In case of intervention on a, use interv_a for generator
            if type(interv_a) == torch.Tensor:
                z_sample_a = torch.cat((z_infer_sample, interv_a), 1)
            else:
                z_sample_a = torch.cat((z_infer_sample, a_train), 1)

            # RECONSTRUCTION LOSS
            # p(x|za)
            x_bin, x_con = self.p_x_za_dist(z_sample_a)
            l1 = x_bin.log_prob(x_train[:, :x_bin_n]).sum(1)  # sum over 19 binary variables
            loss['Reconstr_x_bin'].append(l1.mean().cpu().detach().float())  # mean over batch
            l2 = x_con.log_prob(x_train[:, -x_con_n:]).sum(1)  # sum over 5 continuous variables
            loss['Reconstr_x_con'].append(l2.mean().cpu().detach().float())  # mean over batch
            # p(t|za)
            # In case of intervention on a, use interv_a for generator
            if type(interv_a) == torch.Tensor:
                t = self.p_t_za_dist(z_infer_sample, interv_a)
            else:
                t = self.p_t_za_dist(z_infer_sample, a_train)
            l3 = t.log_prob(t_train).squeeze()
            loss['Reconstr_t'].append(l3.mean().cpu().detach().float())  # mean over batch
            # p(y|t,z,a)
            # In case of intervention on a, use interv_a for generator
            if type(interv_a) == torch.Tensor:
                y = self.p_y_zta_dist(z_infer_sample, t_train, interv_a)
            else:
                y = self.p_y_zta_dist(z_infer_sample, t_train, a_train)

            l4 = y.log_prob(y_train).squeeze()
            loss['Reconstr_y'].append(l4.mean().cpu().detach().float())  # mean over batch

            # REGULARIZATION LOSS
            # approximate Negative KL  p(z) - q(z|x,t,y)
            l5 = (self.p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
            loss['Regularization'].append(l5.mean().cpu().detach().float())  # mean over batch

            # Total objective
            # inner sum to calculate loss per item, torch.mean over batch
            if epoch < args.tStart:
                # force learning t distribution for variance
                loss_mean = torch.mean(l3)
            if args.tStart <= epoch:
                # weighted loss
                factor = 0.1
                loss_mean = torch.mean(l3 + factor * (l1 + l2 + l4 + l5))

            loss['Total'].append(loss_mean.cpu().detach().numpy())

            sample_losses = torch.cat((sample_losses, loss_mean.unsqueeze(0)))
            sample_y_mean.append(y.mean)

        objective = -torch.mean(sample_losses)  # take mean over objective for each z-value

        # average normal distribution means over z_samples, keep batch intact
        y_mean = torch.mean(torch.stack(sample_y_mean), 0)

        return loss, objective, z_infer, y_mean


class CFM_m(nn.Module):
    def __init__(self, z_dim=20, h_dim_f=20, out_dim=1):
        super().__init__()
        # init network
        self.lin1 = nn.Linear(z_dim, h_dim_f)
        self.lin2 = nn.Linear(h_dim_f, out_dim)

    def forward(self, z_in):
        h1 = F.elu(self.lin1(z_in))
        out = F.elu(self.lin2(h1))

        return out
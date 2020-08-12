import numpy as np
import torch
import matplotlib.pyplot as plt


class Evaluator(object):
    def __init__(self, y, a, t, mu):
        # Notation: y01 -> y_{t=0,a=1}
        self.y = y
        self.y00, self.y10, self.y01, self.y11 = [self.y[:, i] for i in range(y.shape[1])]
        self.t0, self.t1 = [t[:, i] for i in range(t.shape[1])]
        self.a = a
        # Notation: mu01 -> mu_{t=0,a=1}
        self.mu00, self.mu10, self.mu01, self.mu11 = [mu[:, i] for i in range(mu.shape[1])]

    def pehe_ty(self, y1a_pred, y0a_pred):
        """PEHE of effect T -> Y"""
        # Select correct ground truth, according to a
        y1a_gt = np.concatenate((self.mu10[:, np.newaxis], self.mu11[:, np.newaxis]), 1)[range(len(y0a_pred)),
                                                                                         self.a[:, 0]]
        y0a_gt = np.concatenate((self.mu00[:, np.newaxis], self.mu01[:, np.newaxis]), 1)[range(len(y0a_pred)),
                                                                                         self.a[:, 0]]
        return np.sqrt(np.mean(np.square((y1a_gt - y0a_gt) - (y1a_pred - y0a_pred))))

    def pehe_at(self, t0_pred, t1_pred):
        """PEHE of effect A -> T"""
        return np.sqrt(np.mean(np.square((self.t1 - self.t0) - (t1_pred - t0_pred))))

    def pehe_ay(self, yt0_pred, yt1_pred):
        """PEHE of effect A -> Y"""
        # use self.t as index to select correct groundtruth
        yt0_gt = np.concatenate((self.mu00[:, np.newaxis], self.mu10[:, np.newaxis]), 1)[range(len(yt1_pred)),
                                                                                         self.t0.astype(int)]
        yt1_gt = np.concatenate((self.mu01[:, np.newaxis], self.mu11[:, np.newaxis]), 1)[range(len(yt1_pred)),
                                                                                         self.t1.astype(int)]
        return np.sqrt(np.mean(np.square((yt1_gt - yt0_gt) - (yt1_pred - yt0_pred))))

    def rmse(self, y_pred, t):
        """rmse of all intervention values, which include factual values"""

        # obtain factual t by indexing on factual a
        fct_t = np.concatenate((self.t0[:, np.newaxis], self.t1[:, np.newaxis]), 1)[range(len(self.t0)), self.a[:, 0]]
        cfct_t = np.concatenate((self.t0[:, np.newaxis], self.t1[:, np.newaxis]), 1)[range(len(self.t0)), 1-self.a[:, 0]]
        # obtain factual index (ta: 00, 10, 01, 11)
        fct_idx = (2*self.a[:, 0] + fct_t).astype('int')

        # factual
        fct_rmse = np.sqrt(np.mean(np.square(y_pred[:, fct_idx] - self.y[:, fct_idx])))
        # counterfactual-a_index: cntra_idx
        cntra_idx = fct_idx.copy()
        cntra_idx[fct_idx == 0], cntra_idx[fct_idx == 1], cntra_idx[fct_idx == 2], cntra_idx[fct_idx == 3] = 2, 3, 0, 1
        cntra_rmse = np.sqrt(np.mean(np.square(y_pred[:, cntra_idx] - self.y[:, cntra_idx])))
        # counterfactual for flipped t
        cntrt_idx = fct_idx.copy()
        cntrt_idx[fct_idx == 0], cntra_idx[fct_idx == 1], cntra_idx[fct_idx == 2], cntra_idx[fct_idx == 3] = 1, 0, 3, 2
        cntrt_rmse = np.sqrt(np.mean(np.square(y_pred[:, cntra_idx] - self.y[:, cntra_idx])))
        # counterfactual for flipped a, t
        cntrta_idx = fct_idx.copy()
        cntrta_idx[fct_idx == 0], cntra_idx[fct_idx == 1], cntra_idx[fct_idx == 2], cntra_idx[fct_idx == 3] = 3, 2, 1, 0
        cntrta_rmse = np.sqrt(np.mean(np.square(y_pred[:, cntra_idx] - self.y[:, cntra_idx])))

        # means squared error of factual: within predicted t values of interventions a=0,1; select one with observed a
        fct_t_rmse = np.sqrt(np.mean(np.square(t[range(len(self.t0)), self.a[:, 0]] - fct_t)))
        cfct_t_rmse = np.sqrt(np.mean(np.square(t[range(len(self.t0)), 1-self.a[:, 0]] - cfct_t)))

        cfct_t_acc = np.sum(np.round(t[range(len(self.t0)), 1-self.a[:, 0]]) == cfct_t)/len(cfct_t)
        fct_t_acc = np.sum(np.round(t[range(len(self.t0)), self.a[:, 0]]) == fct_t)/len(fct_t)
        print('accuracy rounded mean t dist: fct: %f, cfct: %f' %(fct_t_acc, cfct_t_acc))
        print('accuracy only predicting 1: fct: %f, cfct: %f' %(np.sum(fct_t == 1)/len(fct_t), np.sum(cfct_t == 1)/len(cfct_t)))

        return fct_rmse, cntra_rmse, cntrt_rmse, cntrta_rmse, fct_t_rmse, cfct_t_rmse


def get_y_eval(CEVAE, x_input, t_input, a_input, filename):
    """Get evaluation values for effect estimates"""
    a_n = a_input.shape
    t_n = t_input.shape

    # Inference of p(Y|do(t),X,A) for T -> Y
    xa = torch.cat((x_input.float(), a_input.float()), 1)
    z_infer = CEVAE.q_z_xa_dist(xa=xa)

    # use training a
    y1a = CEVAE.p_y_zta_dist(z_infer.mean, torch.ones(t_n).cuda(), a_input.float()).mean.cpu().detach().numpy()
    y0a = CEVAE.p_y_zta_dist(z_infer.mean, torch.zeros(t_n).cuda(), a_input.float()).mean.cpu().detach().numpy()

    # Inference of p(Y|do(a),T,X) for A -> Y
    # a = 0
    xa = torch.cat((x_input.float(), torch.zeros(a_n).cuda()), 1)
    z_infer = CEVAE.q_z_xa_dist(xa=xa)
    t0 = CEVAE.p_t_za_dist(z_infer.mean, torch.zeros(a_n).cuda())
    yt0 = CEVAE.p_y_zta_dist(z_infer.mean, torch.round(t0.mean),  torch.zeros(a_n).cuda()).mean.cpu().detach().numpy()
    # a = 0
    xa = torch.cat((x_input.float(), torch.ones(a_n).cuda()), 1)
    z_infer = CEVAE.q_z_xa_dist(xa=xa)
    t1 = CEVAE.p_t_za_dist(z_infer.mean, torch.ones(a_n).cuda())
    yt1 = CEVAE.p_y_zta_dist(z_infer.mean, torch.round(t1.mean), torch.ones(a_n).cuda()).mean.cpu().detach().numpy()

    t0_std, t1_std = torch.std(t0.mean), torch.std(t1.mean)
    t0, t1 = t0.mean.cpu().detach().numpy(), t1.mean.cpu().detach().numpy()

    return y0a, y1a, t0, t1, t0_std, t1_std, yt0, yt1


def get_intervention_values(CEVAE, x_input, t_input, a_input):
    """Get intervention values for rmse estimates"""
    a_n = a_input.shape
    t_n = t_input.shape

    # a = 0
    xa = torch.cat((x_input.float(), torch.zeros(a_n).cuda()), 1)
    z_infer = CEVAE.q_z_xa_dist(xa=xa)
    # use 'inf' to indicate inferred
    t0_inf = CEVAE.p_t_za_dist(z_infer.mean, torch.zeros(a_n).cuda()).mean.cpu().detach().numpy()
    # t = 0, a = 0
    y00_inf = CEVAE.p_y_zta_dist(z_infer.mean, torch.zeros(t_n).cuda(), torch.zeros(a_n).cuda()).mean.cpu().detach().numpy()
    # t = 1, a = 0
    y10_inf = CEVAE.p_y_zta_dist(z_infer.mean, torch.ones(t_n).cuda(), torch.zeros(a_n).cuda()).mean.cpu().detach().numpy()

    # a = 1
    xa = torch.cat((x_input.float(), torch.ones(a_n).cuda()), 1)
    z_infer = CEVAE.q_z_xa_dist(xa=xa)
    t1_inf = CEVAE.p_t_za_dist(z_infer.mean, torch.ones(a_n).cuda()).mean.cpu().detach().numpy()
    # t = 0, a = 1
    y01_inf = CEVAE.p_y_zta_dist(z_infer.mean, torch.zeros(t_n).cuda(), torch.zeros(a_n).cuda()).mean.cpu().detach().numpy()
    # t = 1, a = 1
    y11_inf = CEVAE.p_y_zta_dist(z_infer.mean, torch.ones(t_n).cuda(), torch.zeros(a_n).cuda()).mean.cpu().detach().numpy()

    return np.concatenate((y00_inf, y10_inf, y01_inf, y11_inf), 1), np.concatenate((t0_inf, t1_inf), 1)


def perform_evaluation(CEVAE, x_input, t_input, a_input, evaluator, pehe, rmse, t_std, filename,
                       loss=None, x_bin_n=None, x_con_n=None, y_input=None):

    # Evaluation
    y0a, y1a, t0, t1, t0_std, t1_std, yt0, yt1 = get_y_eval(CEVAE, torch.tensor(x_input).cuda(),
                                            torch.tensor(t_input).cuda(), torch.tensor(a_input).cuda(), filename)
    pehe_at = evaluator.pehe_at(t0, t1)
    pehe_ty = evaluator.pehe_ty(y1a, y0a)
    pehe_ay = evaluator.pehe_ay(yt0, yt1)
    pehe['A -> T'].append(pehe_at)
    pehe['T -> Y'].append(pehe_ty)
    pehe['A -> Y'].append(pehe_ay)

    print("PEHE -- A->T: %f, T->Y: %f, A->Y: %f" % (pehe_at, pehe_ty, pehe_ay))

    # RMSE evaluation
    y, t = get_intervention_values(CEVAE, torch.tensor(x_input).cuda(),
                                   torch.tensor(t_input).cuda(), torch.tensor(a_input).cuda())

    fct_rmse, cntra_rmse, cntrt_rmse, cntrta_rmse, fct_t, cfct_t = evaluator.rmse(y, t)
    rmse['Y - Factual'].append(fct_rmse)
    rmse['Y - Counter a'].append(cntra_rmse)
    rmse['Y - Counter t'].append(cntrt_rmse)
    rmse['Y - Counter a,t'].append(cntrta_rmse)
    rmse['T - Factual'].append(fct_t)
    rmse['T - Counter'].append(cfct_t)
    t_std['Std[E(T|A=0)]'].append(t0_std)
    t_std['Std[E(T|A=1)]'].append(t1_std)

    print("RMSE y --- factual: %.5f --- counter a: %.5f, t: %.5f, ta: %.5f" % (fct_rmse, cntra_rmse,
                                                                               cntrt_rmse, cntrta_rmse))
    print("RMSE t --- factual: %.5f --- counter: %.5f " % (fct_t, cfct_t))

    # loss is evaluated in test modes
    if loss is not None:
        xa = torch.cat((x_input, a_input), 1)
        z_infer = CEVAE.q_z_xa_dist(xa=xa)
        z_infer_sample = z_infer.mean
        z_inf_a = torch.cat((z_infer_sample, a_input), 1)
        x_bin, x_con = CEVAE.p_x_za_dist(z_inf_a)
        l1 = x_bin.log_prob(x_input[:, :x_bin_n]).sum(1)
        loss['Reconstr_x_bin'].append(l1.mean().cpu().detach().float())
        l2 = x_con.log_prob(x_input[:, -x_con_n:]).sum(1)
        loss['Reconstr_x_con'].append(l2.mean().cpu().detach().float())
        t = CEVAE.p_t_za_dist(z_infer_sample, a_input)
        l3 = t.log_prob(t_input).squeeze()
        loss['Reconstr_t'].append(l3.mean().cpu().detach().float())
        y = CEVAE.p_y_zta_dist(z_infer_sample, t_input, a_input)
        l4 = y.log_prob(y_input).squeeze()
        loss['Reconstr_y'].append(l4.mean().cpu().detach().float())
        l5 = (CEVAE.p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)
        loss['Regularization'].append(l5.mean().cpu().detach().float())
        loss['total'].append(torch.mean(l1 + l2 + l3 + l4 + l5).cpu().detach().numpy())
        return pehe, rmse, t_std, loss

    return pehe, rmse, t_std


def plot_figure(loss, rmse, t_std, pehe, t_start, n_z, n_iter, filename):
    fig = plt.figure(figsize=(19.2, 5), dpi=100)

    # plot objective parts
    subidx = 1
    for key, value in loss.items():
        if subidx < 6:
            plt.subplot(1, 5, subidx)
            plt.plot(np.array(value))
            plt.title(key)
            plt.xlabel('Training time')
            plt.ylabel('Average Loss')
            if n_iter == 1:
                plt.axvspan(0, min(t_start*n_iter, len(np.array(value))), facecolor='b', alpha=0.1, label='Only train T')
            else:
                plt.axvspan(0, min(t_start*n_z*n_iter, len(np.array(value))), facecolor='b', alpha=0.1, label='Only train T')
            plt.legend()
            subidx += 1

    plt.tight_layout()
    fig.savefig('results/' + filename)
    plt.close()

    plt.figure()
    for key, value in t_std.items():
        plt.plot(np.array(value), label=key)
    plt.title("Standard Deviation T distribution")
    plt.xlabel("Training iterations")
    plt.ylabel('std of distribution means in batch')
    plt.axvspan(0, min(t_start*n_z, len(np.array(value))), facecolor='b', alpha=0.1, label='Only train T')
    plt.legend()
    plt.savefig('results/T_dist' + filename)
    plt.close()
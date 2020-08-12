import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
import torch
from torch import optim

from utils import load_data, CEVAE_mod, scatter_latent

parser = ArgumentParser()
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-hDim', type=int, default=50)
parser.add_argument('-nTest', type=int, default=9000)
parser.add_argument('-zDim', type=int, default=5)
parser.add_argument('-rep', type=int, default=20)
parser.add_argument('-nIter', type=int, default=15001)
parser.add_argument('-batchSize', type=int, default=512)
parser.add_argument('-nSamplesZ', type=int, default=1)
parser.add_argument('-evalIter', type=int, default=500)
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-comment', type=str, default='')
args = parser.parse_args()

if __name__ == "__main__":
    acc_list = []
    # repetitions to test stability of training
    for rep_i in range(args.rep):
        # create filename storing models under different settings
        filename = ''
        for key, value in vars(args).items():
            # rewrite repetitions from total to current count
            if key == 'rep':
                value = rep_i
            filename += str(key) + '-' + str(value) + '_'

        # data loader:
        # order of packed data: data = [y, x_con, x_bin, r, b, a]
        train_data, test_data, cat_bin_dict = load_data(n_test=args.nTest)

        # unpack train x in order to init in right shapes
        y_tr, x_tr_con, x_tr_bin, r_tr, b_tr, a_tr = train_data
        x_tr = np.hstack((x_tr_bin, x_tr_con))

        CEVAE = CEVAE_mod(args=args, dim_x=x_tr.shape[1], dim_b=b_tr.shape[1], dim_a=1, dim_x_con=x_tr_con.shape[1],
                          dim_x_bin=cat_bin_dict, dim_z=args.zDim, dim_r=r_tr.shape[1], dim_q_h=args.hDim,
                          dim_p_h=args.hDim).to(args.device)

        # Initialise optimiser ADAM
        optimizer = optim.Adam(CEVAE.parameters(), lr=args.lr)
        # Maintain loss development for monitoring
        loss_dict = defaultdict(list)
        test_loss_dict = defaultdict(list)

        # training loop
        for i in range(args.nIter):
            # select random batch
            # batch size cannot be larger than training data
            assert args.batchSize < x_tr.shape[0]
            batch_idx = np.random.choice(a=range(x_tr.shape[0]), size=args.batchSize, replace=False)
            batch_data = [torch.Tensor(g[batch_idx]).to(args.device) for g in train_data]

            # forward pass
            objective, loss_dict = CEVAE.forward(batch_data, args, loss_dict, cat_bin_dict)
            # Clear optimizer grads
            optimizer.zero_grad()
            # Calculate grads
            objective.backward()
            # Optimizer step
            optimizer.step()

            # Append loss to monitor progress
            loss_dict['train_loss'].append(objective.cpu().detach().numpy())

            # update progress
            if i % args.evalIter == 0:
                print('Iteration: %i / %i ' % (i, args.nIter))
                batch_data = [torch.Tensor(g).to(args.device) for g in test_data]
                accuracy, test_loss, z_mean = CEVAE.forward(batch_data, args, test_loss_dict, cat_bin_dict, eval=True)
                test_loss_dict['test_loss'].append(test_loss.cpu().detach().numpy())
                test_loss_dict['test_loss_idx'].append(i)
                print('Accuracy test set: %f' % accuracy)

                # scatter latent for independence indication
                a_batch = batch_data[-1]
                scatter_latent(z=z_mean, condition=a_batch, progress=(rep_i, i))

                # save models
                torch.save(CEVAE.state_dict(), 'models/CEVAE_' + filename + '.pt')

        # save loss development numerical
        np.save('output/train_loss_develop_' + filename + '.npy', loss_dict['train_loss'])
        np.save('output/test_loss_develop_' + filename + '.npy', test_loss_dict['test_loss'])
        np.save('output/y_train_loss_' + filename + '.npy', loss_dict['Reconstr_y'])
        np.save('output/y_test_loss_' + filename + '.npy', test_loss_dict['Reconstr_y'])

        plt.figure(figsize=(18.0, 12.0))
        plt.plot(loss_dict['train_loss'], label='train')
        plt.plot(test_loss_dict['test_loss_idx'], test_loss_dict['test_loss'], label='test')
        plt.legend()
        plt.title('Loss development training CEVAE')
        plt.tight_layout()
        plt.savefig('output/loss_develop_' + filename + '.png')
        plt.close()

        # save loss development of separate loss components
        plt.figure(figsize=(18.0, 12.0))
        subidx = 1
        for key, value in test_loss_dict.items():
            if key != 'test_loss_idx':
                plt.subplot(2, 4, subidx)
                plt.plot(np.array(value), label=key)
                plt.title(key)
                subidx += 1
        plt.tight_layout()
        plt.savefig('output/test_loss_components_' + filename + '.png')
        plt.close()

        # save loss development of separate loss components
        plt.figure(figsize=(18.0, 12.0))
        subidx = 1
        for key, value in loss_dict.items():
            plt.subplot(2, 4, subidx)
            plt.plot(np.array(value), label=key)
            plt.title(key)
            subidx += 1
        plt.tight_layout()
        plt.savefig('output/loss_components_' + filename + '.png')
        plt.close()

        # append only last accuracy to list
        acc_list.append(accuracy)

    # save list accuracy
    np.save('output/cevae_acc.npy', acc_list)

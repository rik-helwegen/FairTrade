import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


def vis_results(data_results):
    """Create set of histograms with outcomes to compare models"""

    plt.figure(figsize=(5.5, 5))
    binedges = np.linspace(5, 20, 25)
    bincenters = 0.5 * (np.array(binedges[1:]) + np.array(binedges[:-1])) - 0.1

    for key, values in data_results.items():

        if key == 'True Y':
            sub_idx = 1

        if key != 'CPE Y(Z,A,T)' and key != 'True Y':

            if key[:3] == 'CPE':
                key = 'AUX' + key[3:]
            plt.subplot(3, 1, sub_idx)

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

            # plt.subplot(3, 1, 1)
            plt.bar(bincenters - 0.15, mean_a0, width=0.3, color='navy', ecolor='black', yerr=std_a0, label='a=0', alpha=0.6)
            plt.bar(bincenters + 0.15, mean_a1, width=0.3, color='firebrick', ecolor='black', yerr=std_a1, label='a=1', alpha=0.6)
            # plt.title(key)
            if sub_idx == 3:
                plt.xlabel('Y value')
            plt.ylabel('Hist ' + key)
            plt.legend()

            sub_idx += 1

    plt.tight_layout()
    plt.show()


data_out = np.load('pse_cfm_data.npy', allow_pickle='TRUE').item()
vis_results(data_out)

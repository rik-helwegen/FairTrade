import numpy as np
from sklearn.model_selection import train_test_split

# number of variables omitted from data to function as confounder
z_len = 1


class IHDP(object):
    # Collecting data from sensitive generated files (see generateIHDP.py)
    def __init__(self, path_data="datasets/IHDP_sens/csv", replications=10):
        self.path_data = path_data
        self.replications = replications

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_sens_' + str(i + 1) + '.csv', delimiter=',')
            a, t, y, mu, z, x = data[:, 0][:, np.newaxis], data[:, 1:3], data[:, 3:7], data[:, 7:11], \
                                data[:, 11:11+z_len], data[:, 11+z_len:]

            # factual sensitive
            a = a.astype('int')
            # factual treatment
            t_factual = t[range(len(t)), a.squeeze()].astype('int')
            # factual outcome
            y_factual = y[range(len(y)), 2*a.squeeze()+t_factual][:, np.newaxis]

            t_factual = t_factual[:, np.newaxis]
            yield (x, a, t_factual, y_factual), (y, mu, z, t)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_sens_' + str(i + 1) + '.csv', delimiter=',')
            a, t, y, mu, z, x = data[:, 0][:, np.newaxis], data[:, 1:3], data[:, 3:7], data[:, 7:11], \
                                data[:, 11:11+z_len], data[:, 11+z_len:]

            # factual sensitive
            a = a.astype('int')
            # factual treatment
            t_factual = t[range(len(t)), a.squeeze()].astype('int')
            # factual outcome
            y_factual = y[range(len(y)), 2*a.squeeze()+t_factual][:, np.newaxis]
            t_factual = t_factual[:, np.newaxis]

            # select index train valid test
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)

            train = (x[itr], a[itr], t_factual[itr], y_factual[itr]), (y[itr], mu[itr], z[itr], t[itr])
            valid = (x[iva], a[iva], t_factual[iva], y_factual[iva]), (y[iva], mu[iva], z[iva], t[iva])
            test = (x[ite], a[ite], t_factual[ite], y_factual[ite]), (y[ite], mu[ite], z[ite], t[ite])

            yield train, valid, test

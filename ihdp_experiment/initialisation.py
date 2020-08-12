import numpy as np
from torch import optim
import torch
import torch.distributions


def init_qz(qz, a, x):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)

    optimizer = optim.Adam(qz.parameters(), lr=0.001)

    for i in range(50):
        batch = np.random.choice(idx, 1)
        a_train, x_train = torch.cuda.FloatTensor(a[batch]), torch.cuda.FloatTensor(x[batch])
        xa = torch.cat((x_train, a_train), 1)

        z_infer = qz(xa=xa)

        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL, using pz ~ N(0,1)
        KLqp = (-torch.log(z_infer.stddev) + 1/2*(z_infer.variance + z_infer.mean**2 - 1)).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError('KL(pz,qz) contains NaN during init')

    return qz

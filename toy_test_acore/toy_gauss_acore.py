import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_data(sample_size, lower_mu, higher_mu, scale=5, split=False, test_fraction=0.1, seed=77):
    """
    Two dimensional gaussian, where both first and second component (x1) are 1-dim gaussians and the parameter of
    interest is the separation between the two, i.e. theta = abs(mu_1 - mu_2). This makes the parameter identifiable
    even when observed sample size equals 1.
    """
    np.random.seed(seed)

    mu_1 = np.random.uniform(low=lower_mu, high=higher_mu, size=sample_size)
    mu_2 = np.random.uniform(low=lower_mu, high=higher_mu, size=sample_size)
    theta = np.abs(mu_1 - mu_2)

    x_1 = np.random.normal(loc=mu_1, scale=scale)
    x_2 = np.random.normal(loc=mu_2, scale=scale)
    data = np.hstack((theta.reshape(-1, 1),
                      x_1.reshape(-1, 1),
                      x_2.reshape(-1, 1)))

    if split:
        shuffled_idx = np.random.permutation(range(sample_size))
        train_index = shuffled_idx[int(test_fraction * len(shuffled_idx)):]
        test_index = shuffled_idx[:int(test_fraction * len(shuffled_idx))]
        train_set, test_set = data[train_index, :], data[test_index, :]

        assert (len(train_set) + len(test_set)) == len(data)
        return train_set, test_set
    else:
        return pd.DataFrame(data, columns=['theta', 'x1', 'x2'])


def plot_data(train_set, test_set, fig_size=(20, 10)):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=fig_size)

    train_plot = ax[0].scatter(x=train_set[:, 1], y=train_set[:, 2], c=train_set[:, 0])
    ax[0].set(xlabel='x1', ylabel='x2')
    cbar_0 = fig.colorbar(train_plot, ax=ax[0])
    cbar_0.set_label(r'$\theta$', rotation=0)

    test_plot = ax[1].scatter(x=test_set[:, 1], y=test_set[:, 2], c=test_set[:, 0])
    ax[1].set(xlabel='x1', ylabel='x2')
    cbar_1 = fig.colorbar(test_plot, ax=ax[1])
    cbar_1.set_label(r'$\theta$', rotation=0)

    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../../ACORE-LFI/')
sys.path.append('../../ACORE-LFI/acore/')
from utils.functions import train_clf



def generate_data(sample_size, beta_a, beta_b, lower_theta, higher_theta, scale=5, split=False, test_fraction=0.1, seed=77):
    """
    We generate two-dimensional observations, where each coordinate comes from an independent Gaussian. We assume variances are known and equal, 
    while the two means are different for each observation and are constructed to be symmetric around 0. The inferential goal is to build a confidence 
    interval for the separation between the two means of the Gaussians that generate the samples, i.e. theta.
    This makes the parameter identifiable even when observed sample size equals 1.
    See ADA report for exact explanation.
    """
    assert higher_theta > 0
    assert lower_theta >= 0
    assert higher_theta > lower_theta
    
    np.random.seed(seed)

    # sample separation parameter
    gamma = np.random.beta(a=beta_a, b=beta_b, size=sample_size)
    theta = (higher_theta - lower_theta)*gamma + lower_theta

    # sample points symmetrically around 0, given theta
    x_1 = np.random.normal(loc=theta/2, scale=scale)
    x_2 = np.random.normal(loc=-theta/2, scale=scale)

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


def plot_data(data, test_set=None, fig_size=(12, 10)):
    
    if test_set is None:
        ncols = 1
    else:
        ncols = 2
        
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=fig_size)
    
    if test_set is None:
        a = ax
    else:
        a = ax[0]
        
    plot = a.scatter(x=data[:, 1], y=data[:, 2], c=data[:, 0])
    a.set(xlabel='x1', ylabel='x2')
    cbar_0 = fig.colorbar(plot, ax=a)
    cbar_0.set_label(r'$\theta$', rotation=0)
    
    if test_set is not None:
        test_plot = ax[1].scatter(x=test_set[:, 1], y=test_set[:, 2], c=test_set[:, 0])
        ax[1].set(xlabel='x1', ylabel='x2')
        cbar_1 = fig.colorbar(test_plot, ax=ax[1])
        cbar_1.set_label(r'$\theta$', rotation=0)

    plt.show()
    
    
def regression_bias(train, test, plot_title="", fig_size=(12, 10)):
    
    # add column of ones for intercept and separate in x and y (theta)
    train_x = np.hstack(
                        (np.ones(shape=(train.shape[0], 1)), 
                         train[:, 1:])
                       )
    train_y = train[:, 0]
    test_x = np.hstack(
                       (np.ones(shape=(test.shape[0], 1)), 
                        test[:, 1:])
                      )
    test_y = test[:, 0]
    
    # least squares
    coefs = np.linalg.lstsq(a=train_x, 
                            b=train_y, 
                            rcond=None)[0]
    predictions = test_x @ coefs
    
    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    sns.lineplot(x=np.array([np.min(test_y), np.max(test_y)]),
                 y=np.array([np.min(test_y), np.max(test_y)]),
                 ax=ax, label=r"true $\theta$", color="darkcyan")
    sns.scatterplot(x=test_y, y=predictions, ax=ax, color="orange")
    ax.set(xlabel=r'true $\theta$', ylabel=r'$\mathbb{E}[\theta|x]$', title=plot_title)
    plt.legend()
    plt.show()
    
def compute_exact_odds(acore, ):
    
    clf_probs = train_clf(sample_size=acore.b,
                          clf_model=acore.classifier_or,
                          gen_function=acore.model.generate_sample,
                          d=acore.model.d,
                          clf_name=acore.classifier_or_name)
    
    
    
    
    
    
    
    
"""
# NOT IDENTIFIABLE FOR THETA IF THIS IS THE SETUP
mu_1 = np.random.uniform(low=lower_mu, high=higher_mu, size=sample_size)
mu_2 = np.random.uniform(low=lower_mu, high=higher_mu, size=sample_size)
theta = np.abs(mu_1 - mu_2)

x_1 = np.random.normal(loc=mu_1, scale=scale)
x_2 = np.random.normal(loc=mu_2, scale=scale)
data = np.hstack((theta.reshape(-1, 1),
                  x_1.reshape(-1, 1),
                  x_2.reshape(-1, 1)))
"""

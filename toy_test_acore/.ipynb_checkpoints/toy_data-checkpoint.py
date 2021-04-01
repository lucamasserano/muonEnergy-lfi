import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_data(sample_size: int, 
                  test_fraction: float, 
                  x_dims: int = 1, 
                  param_dims: int = 1,
                  slope: float = 10,
                  precision: float = 3,
                  seed: int = 77, 
                  split: bool = True):
    
    if seed is not None:
        np.random.seed(seed)

    theta = np.random.uniform(low=0, high=1, size=(sample_size, x_dims))
    x = np.random.normal(loc=slope*theta, scale=theta/precision, size=(sample_size, param_dims))
    data = np.hstack((theta, x))
    
    if split:
        shuffled_idx = np.random.permutation(range(sample_size))
        train_index = shuffled_idx[int(test_fraction * len(shuffled_idx)):]
        test_index = shuffled_idx[:int(test_fraction * len(shuffled_idx))]
        train_set, test_set = data[train_index, :], data[test_index, :]
        
        assert (len(train_set) + len(test_set)) == len(data)
        return train_set, test_set
    else:
        return pd.DataFrame(data, columns=['theta', 'x'])


def plot_data(train_set, test_set):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    sns.scatterplot(x=train_set[:, 0], y=train_set[:, 1], ax=ax[0])
    sns.scatterplot(x=test_set[:, 0], y=test_set[:, 1], ax=ax[1])
    ax[0].set(xlabel=r'$\theta$', ylabel='x')
    ax[1].set(xlabel=r'$\theta$', ylabel='x')
    plt.show()
    
    
def plot_vectors(vectors, title='VIZ', labels=None, dimensions=3, view_init=(0, 0), fig_size=(10,10)):
    """
    plot the vectors in 2 or 3 dimensions. 
    If labels are supplied, use them to color the data accordingly
    """
    # set up graph
    fig = plt.figure(figsize=fig_size)

    # create data frame
    df = pd.DataFrame(data={'x':vectors[:,0], 'y': vectors[:,1]})
    # add labels, if supplied
    if labels is not None:
        df['label'] = labels
    else:
        df['label'] = [''] * len(df)

    # assign colors to labels
    cm = plt.get_cmap('afmhot') # choose the color palette
    n_labels = len(df.label.unique())
    label_colors = [cm(1. * i/n_labels) for i in range(n_labels)]
    cMap = colors.ListedColormap(label_colors)
        
    # plot in 3 dimensions
    if dimensions == 3:
        # add z-axis information
        df['z'] = vectors[:,2]
        # define plot
        ax = fig.add_subplot(111, projection='3d')
        frame1 = plt.gca() 
        # remove axis ticks
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])
        frame1.axes.zaxis.set_ticklabels([])

        # plot each label as scatter plot in its own color
        for l, label in enumerate(df.label.unique()):
            df2 = df[df.label == label]
            color_values = [label_colors[l]] * len(df2)
            ax.scatter(df2['x'], df2['y'], df2['z'], 
                       c=color_values, 
                       cmap=cMap, 
                       edgecolor=None, 
                       label=label, 
                       alpha=0.4, 
                       s=100)
      
    # plot in 2 dimensions
    elif dimensions == 2:
        ax = fig.add_subplot(111)
        frame1 = plt.gca() 
        frame1.axes.xaxis.set_ticklabels([])
        frame1.axes.yaxis.set_ticklabels([])

        for l, label in enumerate(df.label.unique()):
            df2 = df[df.label == label]
            color_values = [label_colors[l]] * len(df2)
            ax.scatter(df2['x'], df2['y'], 
                       c=color_values, 
                       cmap=cMap, 
                       edgecolor=None, 
                       label=label, 
                       alpha=0.4, 
                       s=100)
    else:
        raise NotImplementedError()

    #for ii in range(0,360,1):
     #   ax.view_init(elev=10., azim=ii)
      #  plt.draw()
        #plt.savefig("./graph/movie%d.png" % ii)
    
    ax.view_init(*view_init)
    plt.title(title)
    plt.legend()
    plt.show()


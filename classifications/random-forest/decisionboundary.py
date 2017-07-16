import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcmp
from matplotlib.colors import cnames
from matplotlib import markers

marker_list_all = list (markers.MarkerStyle.markers.keys())
random.shuffle (marker_list_all)
color_list_all = list (cnames)
random.shuffle (color_list_all)
color_list_all2 = list (cnames)
random.shuffle (color_list_all2)


def plot_decision_boundary (x, y, classifier,
                            feature_ind1=0, feature_ind2=1,
                            standardizer=None,
                            num=150, color_list=None,
                            marker_list=None, outfile_name='class_bd.pdf'):
    # a function for two variable classification visualization
    # classifier must be instantiated outside the function
    # let's first store some colors
    num_classes = len (np.unique (y))
    if color_list!=None:
        assert len (color_list) >= num_classes
    else:
        color_list = color_list_all[:num_classes]
        
    # Now the markers for scatter plot
    if marker_list is not None:
        assert len (marker_list) >= num_classes
    else:
        marker_list = marker_list_all[:num_classes]
        
    if standardizer is not None:
        x_std = standardizer.transform (x)
    else:
        x_std = x;
    x1min, x1max = x_std[:,feature_ind1].min(), x_std[:,feature_ind1].max()
    x2min, x2max = x_std[:,feature_ind2].min(), x_std[:,feature_ind2].max()
    x1std, x2std = np.meshgrid (np.linspace (x1min, x1max, num),
                                np.linspace (x2min, x2max, num))
    ypd = classifier.predict (np.array ([x1std.ravel (), x2std.ravel ()]).T)
    ypd=np.reshape (ypd, np.shape (x1std))
    print ('shape: ', np.shape (ypd), ', type: ', type(ypd))
    xx1, xx2 = np.meshgrid (np.linspace (x[:,feature_ind1].min(), x[:,feature_ind1].max(), num),
                            np.linspace (x[:,feature_ind2].min(), x[:,feature_ind2].max(), num))
    cmp=lcmp(color_list[:num_classes])
    plt.contourf (xx1, xx2, ypd, cmap=cmp, alpha=0.1)
    for ind, val in enumerate (np.unique(y)):
        plt.scatter (x[y==val,0], x[y==val,1],
                     c=color_list_all2[ind], marker=marker_list[ind],
                     label='Class '+str(val), alpha=1)
    plt.xlabel ('Feature ' + str(feature_ind1))
    plt.ylabel ('Feature ' + str(feature_ind2))
    plt.xlim (x[:,feature_ind1].min(), x[:,feature_ind1].max())
    plt.ylim (x[:,feature_ind2].min(), x[:,feature_ind2].max())
    plt.legend(loc='best')
    plt.tight_layout ()
    plt.savefig (outfile_name)
    plt.show ()

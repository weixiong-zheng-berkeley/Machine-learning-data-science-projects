import numpy as np
import matplotlib.pyplot as plt
# svm functionality
from sklearn.tree import DecisionTreeClassifier as DTC
# data set needed
from sklearn import datasets as ds
# data set separation
from sklearn.model_selection import train_test_split as tts

# plotting results requirement
from decisionboundary import plot_decision_boundary as pdb
from decisionboundary import marker_list_all, color_list_all
# standardizing data
from sklearn.preprocessing import StandardScaler as SC
# get the data from iris
iris = ds.load_iris ()
x = iris.data[:,[2,3]]
y = iris.target

# visualize how they look
num_classes = len(np.unique(y))
## class 1
for ind, val in enumerate (np.unique(y)):
    plt.scatter (x[y==val,0], x[y==val,1],
                 marker = marker_list_all[ind],
                 c = color_list_all[ind],
                 label='Class '+str(val))
plt.legend (loc = 0)
plt.xlim (x[:,0].min(), x[:,0].max())
plt.ylim (x[:,1].min(), x[:,1].max())
plt.tight_layout ()
pic1 = 'scatter-show.pdf'
plt.savefig (pic1)
plt.show ()

# separating data set
xtr, xte, ytr, yte = tts (x, y, test_size = 0.3)
# standarizing the data
sc0 = SC ()
sc0.fit (xtr)
xtr_std = sc0.transform (xtr)
xte_std = sc0.transform (xte)
# The following is for classifying
dtc =  DTC()
dtc.fit (xtr_std, ytr)
ypd = dtc.predict (xte_std)
print ("accuracy: ", dtc.score (xte_std, yte))

pdb (x, y, classifier=dtc, standardizer=sc0)

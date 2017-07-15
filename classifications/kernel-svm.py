import numpy as np
#import matplotlib.pyplot as plt
# svm functionality
from sklearn.svm import SVC
# data set separation
from sklearn.model_selection import train_test_split as tts
from decisionboundary import plot_decision_boundary as pdb
from decisionboundary import marker_list_all, color_list_all
xxor = np.random.randn(350,2)
yxor = np.logical_xor(xxor[:,0]>0, xxor[:,1]>0)
# visualize how they look
## class 1
plt.scatter (xxor[yxor==False, 0], xxor[yxor==False, 1],
             marker=marker_list_all[0], c=color_list_all[0], label='Class False')
## class 1
plt.scatter (xxor[yxor==True, 0], xxor[yxor==True, 1],
             marker=marker_list_all[1], c=color_list_all[1], label='Class True')
plt.legend (loc = 0)
plt.xlim (xxor[:,0].min(), xxor[:,0].max())
plt.ylim (xxor[:,1].min(), xxor[:,1].max())
plt.tight_layout ()
pic1 = 'scatter-show.pdf'
plt.savefig (pic1)
plt.show ()

# separating data set
xtr, xte, ytr, yte = tts (xxor, yxor, test_size = 0.3)
# The following is for classifying
svc0 = SVC (C=100.0, kernel='rbf')
svc0.fit (xtr, ytr)
ypd = svc0.predict (xte)
print ("accuracy: ", svc0.score (xte, yte))

pdb (xxor, yxor, classifier=svc0)

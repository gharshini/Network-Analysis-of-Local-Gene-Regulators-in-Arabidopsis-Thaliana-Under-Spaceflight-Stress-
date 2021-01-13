import pandas as pd 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsClassifier
import scipy as sp
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from pandas import ExcelWriter 
from pandas import ExcelFile
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn import metrics


data_frame = pd.read_csv("ws-GC-m.csv")

print(data_frame.head(5))

genes=data_frame.loc[:,['Genes']].values

print(genes)

features = ['sub-cen', 'clo-cen', 'degree', 'katz', 'p-rank']
# Separating out the features
x = data_frame.loc[:, features].values
# Separating out the classes
y = data_frame.loc[:,['classes']].values
# Standardizing the features

print(y)

data_sacaled=scale(x, axis=0, with_mean=True, with_std=True, copy=True)
print(data_sacaled)
results = pd.DataFrame(data_sacaled)
results.columns = ['sub-cen', 'clo-cen', 'degree', 'katz', 'p-rank']
results.insert(5, "classes", y, True) 


print(results)

##############################logistic regrassion ###################################

'''labels=results.iloc[:,5]
X_train,X_test,y_train,y_test=train_test_split(data_sacaled,labels,test_size=0.00000000001, random_state=42)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(X_train))]
print ('training data',X_train)
print ('')'''

LOG= LogisticRegression(random_state=0).fit(data_sacaled, y)

pre=LOG.predict_proba(data_sacaled)





probabilities=pd.DataFrame(pre)
#probabilities.insert(1,'Genes',genes,True)

print(probabilities)

pro_results=pd.Series(np.diag(probabilities))


print('probabilities',pro_results)

writer = ExcelWriter('pro_ws-GC.xlsx')
pro_results.to_excel(writer,'Sheet1',index=False)
writer.save()


#pre = pre[:, 1]


print('probabilidad',np.transpose(pre))

test=np.array([0, 5,30,13,34,55])
prob=np.array([0.00184997,0.00814319,0.00582494,0.00142175,0.0001452 ,0.00903733])
print('test',test)
# calculate scores
#ns_auc = roc_auc_score(y_test, ns_probs, multi_class='ovo')
lr_auc = roc_auc_score(y_test, np.transpose(pre), multi_class='ovo')
# summarize scores
#print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves

lr_fpr, lr_tpr, _ = roc_curve(y_test, pre)
#ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

# plot the roc curve for the model
#pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
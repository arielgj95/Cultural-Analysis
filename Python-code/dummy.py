import pandas as pd
import json,os
from sklearn.preprocessing import LabelEncoder
#from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, make_scorer
import matplotlib.pyplot as plt

from random import sample
import numpy as np
import sys

path = r'C:\users\ariel\Documents\PHD\osfstorage-archive\code\svc\new2'
os.chdir(path)
Data = pd.read_csv(os.path.join(path,'All_Foodpictures_information_13.csv'))
Data = pd.concat([Data['Average_Valence_UK'], Data['Average_Valence_US'], Data['Average_Valence_JP'],
                  Data['Food classification'], Data.iloc[:,8:30]], axis=1)
Data = Data.dropna(axis=0)
#print(len(Data.loc[Data['Average_Arousal_JP']>50]))

Data_Asian = Data[Data['Food classification']=='Asian'].drop('Food classification', axis=1)
Data_Western = Data[Data['Food classification']=='Western'].drop('Food classification', axis=1)
Data_Asian = Data_Asian.to_numpy()
Data_Western = Data_Western.to_numpy()
Data = Data.to_numpy()
seed = 10
rng = np.random.RandomState(seed)
indices = rng.randint(0, len(Data_Western), len(Data_Western))

#print(Data_Asian[:,0],Data_Asian[:,0][indices])

#print(len(Data_Western),len(Data_Asian))
avg_arousals=[]

labels = [f'{f}-{f + 5}' for f in range(0,100,5)]
ticks = range(len(labels))
countries = ["UK", "US", "JP"]

for i in range(3):
    arousals_as_we = []
    indices = rng.randint(0, len(Data_Western), len(Data_Western))
    arousals_as_we.extend(Data_Asian[:,i][indices])
    arousals_as_we.extend(Data_Western[:,i][indices])
    avg_arousals.append(round(np.mean(arousals_as_we)))
    counts = []
    plt.figure(figsize = (15, 9))
    #print(f"Data1: {Data[:, i]} \n")
    for f in range(0,100,5):
        counts.append(np.count_nonzero((Data[:,i]>f) & (Data[:,i]<f+5)))
    #print(counts)
    plt.bar(ticks, counts, align='center')
    plt.title(f"Valence of {countries[i]}")
    plt.xticks(ticks, labels)
    plt.savefig(f"Valence of {countries[i]}")
    #plt.show()
    plt.clf()
print(avg_arousals)

#print(Data_Asian[:,0:3])
'''
Data_Asian[:,0:3] = np.where(Data_Asian[:,0:3] > 50, 1, 0)          #Arousal = 1 if >50, 0 otherwise
Data_Western[:,0:3] = np.where(Data_Western[:,0:3] > 50, 1, 0)
#print((Data_Western[:,2]==1).sum())
np.set_printoptions(threshold=sys.maxsize)
#print(len(Data_Asian), len(Data_Western), Data_Asian[:,3][0:10])
'''



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

def json_data(file):
    if os.path.exists("{}".format(file)):
        with open("{}".format(file), "r", encoding="utf-8") as js:
            j_data = json.load(js)
            if j_data != '':
                CM = np.array(j_data['CM'])
                n_it = j_data['n_it']
                js.close()
            else:
                CM = np.zeros([4, 2, 2, 2])     #3 models each tested with 2 test sets (6 CF). Each CF is 2x2 because there are 2 labels
                n_it = 0
    else:
        CM = np.zeros([4, 2, 2, 2])
        n_it = 0
    return CM, n_it

path = r'C:\users\ariel\Documents\PHD\osfstorage-archive\code\svc\new2'
os.chdir(path)

Data = pd.read_csv(os.path.join(path,'All_Foodpictures_information_13.csv'))
Data = pd.concat([Data['Sweet vs savory'], Data['Food classification'], Data.iloc[:,8:30]], axis=1)
Data = Data[(Data['Sweet vs savory']=='Sweet') | (Data['Sweet vs savory']=='Savory')]
Data = Data.dropna(axis=0)


label_encoder = LabelEncoder()
Data['Sweet vs savory'] = label_encoder.fit_transform(Data['Sweet vs savory'])

Data_Asian = Data[Data['Food classification']=='Asian'].drop('Food classification', axis=1)
Data_Western = Data[Data['Food classification']=='Western'].drop('Food classification', axis=1)
print(len(Data_Asian['Sweet vs savory']=='Sweet'))
Data_Asian = Data_Asian.to_numpy()
Data_Western = Data_Western.to_numpy()
np.set_printoptions(threshold=sys.maxsize)

n_mc = 30
n_bstrap = 500
seed = 10
rng = np.random.RandomState(seed)
#n_trees = 100
n_tr = 190
C_All = np.zeros([n_mc,4, 2, 2, 2]) #it contains all the possible confusion matrices (used to compute mean and std)
print(C_All[0])
scalerX = preprocessing.MinMaxScaler()
Data_Asian = scalerX.fit_transform(Data_Asian)
Data_Western = scalerX.fit_transform(Data_Western)
#used for ROC: 4 models tested with asian and western data, each model has been tested with x_true thus obtaining prob
all_predictions = [[[[] for i in range(2)] for j in range(2)] for k in range(4)]
#print("all predictions:",np.asarray(all_predictions).shape,"\n", all_predictions)
models = ["Western", "Asian", "Western + Asian","Western + Asian + FC labels"]
tests = ["Asian", "Western"]
classes = ["Savory", "Sweet"]
#weights = np.linspace(0.0, 1.0, 20)


'''
grid = {'n_estimators': [n_trees],          #gamma, lambda 10^-4, 10^-2 log scale
        'max_features': [1, 2, 4, 5, 6, 8, 10]} #usa balanced per class weight, no max features, usa SVM
        #'class_weight': [{0:0.2, 1:0.8}]} #[{0: x, 1: 1.0 - x} for x in weights]}
'''
print(np.array(all_predictions).shape)
grid = {'C':        np.logspace(-4, 3 , 30),
        'kernel':   ['rbf'],
        'gamma':    np.logspace(-4, 3 , 30),
        #'class_weight': [{0: x, 1: 1.0 - x} for x in weights]}
        'class_weight': ['balanced']}

kappa_scorer = make_scorer(cohen_kappa_score)
mcc = make_scorer(matthews_corrcoef)
#all_scorers = [kappa_scorer, mcc, 'balanced_accuracy', 'f1']

#CM, n_it = json_data("CM_ba_tmp.json")
n_it=0
CM = np.zeros([4, 2, 2, 2])

for i in range(n_it, n_mc):

    n = Data_Asian.shape[0]
    j = sample(range(n), n)
    XTr_Asian = Data_Asian[j[0:n_tr], 1:]
    #XTr_Asian= scalerX.fit_transform(XTr_Asian) #preprocess
    YTr_Asian = Data_Asian[j[0:n_tr], 0]
    XTe_Asian = Data_Asian[j[n_tr:n], 1:]
    print(XTe_Asian.shape)
    #XTe_Asian = scalerX.transform(XTe_Asian)
    YTe_Asian = Data_Asian[j[n_tr:n], 0]

    n = Data_Western.shape[0]
    j = sample(range(n), n)
    XTr_Western = Data_Western[j[0:n_tr], 1:]
    #XTr_Western = scalerX.fit_transform(XTr_Western)  # preprocess
    YTr_Western = Data_Western[j[0:n_tr], 0]
    XTe_Western = Data_Western[j[n_tr:n], 1:]
    #XTe_Western = scalerX.transform(XTe_Western)
    YTe_Western = Data_Western[j[n_tr:n], 0]

    best_params = []
    best_score = []

    for j in range(4):

        if j == 0:
            XTr = XTr_Western
            YTr = YTr_Western

        if j == 1:
            XTr = XTr_Asian
            YTr = YTr_Asian

        if j == 2:
            XTr = np.concatenate([XTr_Western, XTr_Asian], axis=0)
            YTr = np.concatenate([YTr_Western, YTr_Asian], axis=0)

        if j == 3:
            XTr_Asian[:, -1] = np.zeros(len(XTr_Asian))
            XTr_Western[:, -1] = np.ones(len(XTr_Western))
            XTe_Asian[:, -1] = np.zeros(len(XTe_Asian))
            XTe_Western[:, -1] = np.ones(len(XTe_Western))
            XTr = np.concatenate([XTr_Western, XTr_Asian], axis=0)
            YTr = np.concatenate([YTr_Western, YTr_Asian], axis=0)


        MS = GridSearchCV(estimator=SVC(),
                          param_grid=grid,
                          #scoring='roc_auc',
                          scoring='balanced_accuracy',
                          cv=10,
                          verbose=0,
                          n_jobs = -1)
        '''
        MS = GridSearchCV(estimator=RandomForestClassifier(),
                          param_grid=grid,  # controllare che il minimo non sia ai bordi
                          scoring='balanced_accuracy',  # testare balanced_accuracy, mcc, f1, k
                          cv=10,
                          verbose=0,
                          n_jobs = -1)
        '''

        H = MS.fit(XTr, YTr)
        best_params.append(H.best_params_)
        best_score.append(H.best_score_)

        '''
        Model = RandomForestClassifier(n_estimators=H.best_params_['n_estimators'],
                                       max_features=H.best_params_['max_features'],
                                       class_weight=H.best_params_['class_weight'])
        '''

        Model = SVC(C=H.best_params_['C'],
                kernel=H.best_params_['kernel'],
                gamma=H.best_params_['gamma'],
                class_weight= H.best_params_['class_weight'],
                probability=True)

        Model.fit(XTr, YTr)

        Y_Asian = Model.predict(XTe_Asian)
        Y_Western = Model.predict(XTe_Western)
        prob_asian = Model.predict_proba(XTe_Asian)
        prob_western = Model.predict_proba(XTe_Western)
        #print("probabilities:",prob_asian, prob_western,len(prob_asian),len(prob_western))
        #Y_realAs_tot = np.concatenate(Y_real_tot, Y_Asian)

        C_All[i][j][0] = confusion_matrix(YTe_Asian,Y_Asian)    #i iteration, j model,c for 0 Asian 1 Western, k rows l columns
        C_All[i][j][1] = confusion_matrix(YTe_Western, Y_Western)

        CM[j][0] = CM[j][0] + C_All[i][j][0]
        CM[j][1] = CM[j][1] + C_All[i][j][1]


        all_predictions[j][0][0].extend(YTe_Asian)
        all_predictions[j][0][1].extend(prob_asian.tolist())
        all_predictions[j][1][0].extend(YTe_Western)
        all_predictions[j][1][1].extend(prob_western.tolist())


        indices = rng.randint(0, len(all_predictions[j][0][0]), len(all_predictions[j][0][0]))
        #print(indices)
        #print(all_predictions[j][0][0], np.asarray(all_predictions[j][0][0])[indices], np.asarray(all_predictions[j][0][1])[:,1][indices])

        #prob=all_predictions[j][0][1]
        #print(np.asarray(prob).shape)
        #print(len(all_predictions[j][0][1][:,1]))
        #print(np.asarray(all_predictions[j][0][1], dtype='object'))
        #print("all predictions:",np.asarray(all_predictions).shape)

        #print(f"CM {j}: \n {CM[j][0]} \n \n {CM[j][1]} \n\n\n")
        #print(f"C_All {i} {j}: \n {C_All[i][j][0]} \n \n {C_All[i][j][1]} \n\n\n")

        for c in range(2):
            print("Confusion matrix by testing a {} model with {} test data:".format(models[j], tests[c]))
            print("CM {} {} in it {}:\n ".format(j,c,i),CM[j][c])

            #sns.plot(values_format = '.2%')
            if i % 10 == 9:
                bootstrap_scores = []
                if j==0 and c==0:
                    fig, axs = plt.subplots(len(tests),len(models), figsize=(18,10))
                #for r in range(2):
                n_fpr, n_tpr, th = roc_curve(all_predictions[j][c][0], np.asarray(all_predictions[j][c][1])[:,1])
                n_auc = roc_auc_score(all_predictions[j][c][0], np.asarray(all_predictions[j][c][1])[:,1])
                #skip=0
                ##############################
                for bn in range(n_bstrap):
                    # bootstrap by sampling with replacement on the prediction indices
                    indices = rng.randint(0, len(all_predictions[j][c][1]), len(all_predictions[j][c][1]))
                    if len(np.unique(np.asarray(all_predictions[j][c][0])[indices])) == 1:
                        #skip=skip+1
                        continue
                    #print(skip)
                    score = roc_auc_score(np.asarray(all_predictions[j][c][0])[indices], np.asarray(all_predictions[j][c][1])[:,1][indices])
                    bootstrap_scores.append(score)
                    # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

                avg_auc = np.mean(bootstrap_scores)
                std_auc = np.std(bootstrap_scores)
                ###########################
                # plot the roc curve for the model
                ns_probs = [0 for _ in range(len(all_predictions[j][c][0]))]
                print(f'Model {models[j]} tested with {tests[c]}, class {classes[1]}: ROC AUC=%.3f' % (n_auc))
                ns_fpr, ns_tpr, _ = roc_curve(all_predictions[j][c][0], ns_probs)
                axs[c, j].plot(n_fpr, n_tpr, marker='.',label=f'SVC')
                axs[c, j].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
                axs[c, j].set_title(f'Model {models[j]} \n tested with {tests[c]} data',fontsize=12)
                #plt.plot(n_fpr, n_tpr, marker='.',label=f'SVC')
                axs[c, j].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
                axs[c, j].text(+0.9, +0.05, "AUC: {:.3f} \n Avg AUC: {:.3f} \n Std AUC: {:.3f}".format(n_auc,avg_auc,std_auc),
                               horizontalalignment='center',
                               verticalalignment='center',
                               fontstyle = 'italic',
                               color = 'darkred',
                               fontsize = 'small')
                                #transform=ax.transAxes)
                axs[c, j].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
                #axs[c, j].label_outer()
                # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
                    # axis labels
                #axs[c, j].xlabel('False Positive Rate')
                #axs[c, j].ylabel('True Positive Rate')
                    # show the legend
                    # show and save the plot
                    #plt.show()
                #plt.savefig(f"ROC curve of {models[j]} model with {tests[c]} test data and class {classes[1]}_it {i}_ba.png")
                if j==3 and c==1:
                    fig.suptitle('ROC curves')
                    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.7)
                    #plt.legend()
                    fig.tight_layout()
                    fig.savefig(f"SvsS ROC curves_it {i}_ba.png")
                    fig.clf()  # clear figure to avoid overlapping

                # creates 4 arrays, each one containing the values obtained in a index of the CM evaluated now
                if j==0 and c==0:
                    fig2, axs2 = plt.subplots(len(tests),len(models),figsize=(18,10))

                all_cm_values = np.array([[x[j][c][k][l]/np.sum(x[j][c]) for x in C_All[0:i + 1]] for k in range(2) for l in range(2)])
                #print(all_cm_values)

                # compute mean and std for each index of CM
                #means = np.mean(all_cm_values, axis=1)
                stds = np.std(all_cm_values, axis=1)
                #print(CM[j][c].flatten() / np.sum(CM[j][c]))
                CM_prc = ["{0: .2%}".format(value) for value in CM[j][c].flatten() / np.sum(CM[j][c])]
                                              
                # create the cells for the CM and print it
                CM_cells = [f"{v1}\n{v2}" for v1, v2  in
                            zip(CM_prc,["std: {0: .2%}".format(value) for value in stds])]

                CM_cells = np.asarray(CM_cells).reshape(2, 2)

                axs2[c, j] = sns.heatmap(CM[j][c], annot=CM_cells, cmap='Blues', fmt="", xticklabels=classes, yticklabels=classes, ax=axs2[c,j])
                axs2[c, j].set_title(f'CM of Model {models[j]} \n tested with {tests[c]} data', fontsize=12)
                if j==3 and c==1:
                    fig2.suptitle('Confusion Matrices')
                    fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.7)
                    #plt.legend()
                    fig2.tight_layout()
                    fig2.savefig("SvsS Confusion matrices_it {}_ba.png".format(i))
                    fig2.clf()  # clear figure to avoid overlapping

            #plt.show()
    '''
    with open("CM_ba_tmp.json", "w", encoding="utf-8") as js:
        j_data = {'CM': CM.tolist(), "n_it": i}
        json.dump(j_data, js)  # write the new confusion matrix
        js.close()

    if i % 5 == 4:
        with open("ba_scores.json", "a", encoding="utf-8") as js:
            j_data = {'order': ['Western','Asian','Western + Asian'], 'scores': best_score , 'params': best_params, "n_it": i}
            json.dump(j_data, js)  # write the new data
            js.close()

    if i == 29:
        with open("CM_ba.json", "w", encoding="utf-8") as js:
            j_data = {'CM': CM.tolist(), "n_it": i}
            json.dump(j_data, js)  # write the new confusion matrix
            os.remove("CM_ba_tmp.json")
            js.close()
    '''


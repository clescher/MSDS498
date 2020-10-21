# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:49:55 2020

@author: clesc
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import statsmodels.api as sms
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE

import time
start_time = time.time()
#may need below line to get imblearn to work
#conda install -c conda-forge imbalanced-learn

os.chdir("C:/Users/clesc/OneDrive/Documents/Northwestern/MSDS 498")
df =  pd.read_csv('modelingdftrain.csv', sep=','  , engine='python')

#function to return accuracy measure & confusion matrix, just enter in y predictions since y_test values don't change
def graph_metrics(y_pred, title, model):
    #confusion matrix that's been normalized
    fig = plt.figure( )
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'],normalize=True)
    sns.heatmap(confusion_matrix, annot=True)
    plt.title(title)
    plt.xticks(np.arange(0.5,2), ['Paid', 'Default'])
    plt.yticks(np.arange(0.5,2), ['Paid', 'Default'])
    plt.show()
    fig.savefig(title + " Con Mat", bbox_inches='tight', dpi=250)

    #roc
    fig = plt.figure( )
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="ROC, auc="+str(auc))
    plt.legend(loc=4)
    plt.title(title+ " ROC")
    plt.show()
    fig.savefig(title + " ROC", bbox_inches='tight', dpi=250)

    #gives us Accuracy Rsquared, MAE, MSE, RMSE and more    
    explained_variance=metrics.explained_variance_score(y_test, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred) 
    mse=metrics.mean_squared_error(y_test, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
    r2=metrics.r2_score(y_test, y_pred)    
    
    return [title, metrics.accuracy_score(y_test, y_pred), round(explained_variance,4),
                         round(mean_squared_log_error,4), round(r2,4), round(mean_absolute_error,4), 
                         round(mse,4), round(np.sqrt(mse),4), auc, 
                         confusion_matrix[0][0], confusion_matrix[1][1], confusion_matrix[0][1], confusion_matrix[1][0]]
    

def cols_used(model):
    d = []
    for i in range(X_train.shape[1]):
        d.append(
            {
                'Feature': X_train.columns[i],
                'Used': model.support_[i],
                'Rank':  model.ranking_[i]
                }
            )
    return pd.DataFrame(d)

#this is our comparison dataframe to judge all models on
#True Negative is when we accurately predict someone will pay off their loan
#True Pos is when we accurately predict someone will default
#False Neg is when we believe someone will pay off their loan but they don't
#False Pos is when we believe someone will default but they don't
resultcols = ['Model', 'Accuracy', 'explained_variance','mean_squared_log_error', 'r2', 'MAE', 'MSE', 'RMSE',
              'AUC','True_Neg', 'True_Pos', 'False_Neg', 'False_Pos']
resultsdf = pd.DataFrame(columns=resultcols)


#this is the full modeling file once we fix the data issues
X = df[['loan_amnt_cuberoot', 'term', 'int_rate_log', 'grade', 
              'emp_length2', 'home_ownership2', 'annual_inc', 
              'desc2', 'purpose', 'dti', 'delinq_2yrs', 
              'revol_util', 'initial_list_status2', 'application_type', 
              'tot_coll_amt', 'chargeoff_within_12_mths', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 
              'total_bal_ex_mort', 'delinq_amt_pct', 
              'sats_pct', 'max_fico_low','len_credit_cuberoot' ]]


#target value
y = df['default_ind']


#process to convert variables to codes from categorical
#might consider putting this into the data prep master
convertlist = ['purpose', 'term', 'application_type', 'grade']
for c in range(0,len(convertlist)):
    X[convertlist[c]] = X[convertlist[c]].astype('category')
    X[convertlist[c]] = X[convertlist[c]].cat.codes


#split the data into test and train sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=13)

#create smote train sets
sm = SMOTE(random_state=13)
columns = X_train.columns
X_train_SMOTE,y_train_SMOTE = sm.fit_resample(X_train, y_train)
X_train_SMOTE = pd.DataFrame(data=X_train_SMOTE,columns=columns )
y_train_SMOTE= pd.DataFrame(data=y_train_SMOTE,columns=['default_ind'])

#show how smote works
print("Number of original no default",len(y_train[y_train==0]))
print("Number of original default",len(y_train[y_train==1]))
print("Proportion of original no default is ",len(y_train[y_train==0])/len(X_train))
print("Proportion of original default is ",len(y_train[y_train==1])/len(X_train))

print("length of oversampled data is ",len(X_train_SMOTE))
print("Number of no default",len(y_train_SMOTE[y_train_SMOTE['default_ind']==0]))
print("Number of default",len(y_train_SMOTE[y_train_SMOTE['default_ind']==1]))
print("Proportion of no default is ",len(y_train_SMOTE[y_train_SMOTE['default_ind']==0])/len(X_train_SMOTE))
print("Proportion of default is ",len(y_train_SMOTE[y_train_SMOTE['default_ind']==1])/len(X_train_SMOTE))

#Logistic regressions with/without smote
#without SMOTE
model1 = LogisticRegression(max_iter=10000)
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)


#reuse this line to append the results of your model to the results df 
#replace the following
#y_pred1 -> results of your predictions
#'Log reg' -> the title of the graphs and model type
#model1 -> what the model is stored as
resultsdf=resultsdf.append(pd.Series(graph_metrics(y_pred1,'Log Reg', model1), index= resultcols), ignore_index=True)
model1.coef_

#with  SMOTE
model1_SMOTE = LogisticRegression(max_iter=10000)
model1_SMOTE.fit(X_train_SMOTE,y_train_SMOTE)
y_pred1_SMOTE=model1_SMOTE.predict(X_test)

resultsdf=resultsdf.append(pd.Series(graph_metrics(y_pred1_SMOTE,'Log Reg SMOTE', model1_SMOTE), index= resultcols), ignore_index=True)
model1_SMOTE.coef_


#RFE
model2 = LogisticRegression(max_iter=10000)
rfe = RFECV(model2, n_jobs=10)
rfe.fit(X_train,y_train)
y_pred2=rfe.predict(X_test)

resultsdf=resultsdf.append(pd.Series(graph_metrics(y_pred2,'Log Reg RFE', rfe), index= resultcols), ignore_index=True)
rfe_vars = cols_used(rfe)
    
#RFE with SMOTE
model2_SMOTE = LogisticRegression(max_iter=10000)
rfe_SMOTE = RFECV(model2_SMOTE, n_jobs=10)
rfe_SMOTE.fit(X_train_SMOTE,y_train_SMOTE)
y_pred2_SMOTE=rfe_SMOTE.predict(X_test)

resultsdf=resultsdf.append(pd.Series(graph_metrics(y_pred2_SMOTE,'Log Reg RFE SMOTE', rfe_SMOTE), index= resultcols), ignore_index=True)
rfe_SMOTE_vars = cols_used(rfe_SMOTE)


#decision tree
model3 = DecisionTreeClassifier()
model3 = model3.fit(X_train,y_train)
y_pred3 = model3.predict(X_test)

resultsdf=resultsdf.append(pd.Series(graph_metrics(y_pred3,'Tree', model3), index= resultcols), ignore_index=True)


#tree with SMOTE
model3_SMOTE = DecisionTreeClassifier()
model3_SMOTE = model3_SMOTE.fit(X_train,y_train)
y_pred3_SMOTE = model3_SMOTE.predict(X_test)

resultsdf=resultsdf.append(pd.Series(graph_metrics(y_pred3_SMOTE,'Tree SMOTE', model3_SMOTE), index= resultcols), ignore_index=True)


#RFEtree
model4 = DecisionTreeClassifier()
rfe2 = RFECV(model4, n_jobs=10)
rfe2.fit(X_train,y_train)
y_pred4=rfe2.predict(X_test)

resultsdf=resultsdf.append(pd.Series(graph_metrics(y_pred4,'Tree RFE', rfe2), index= resultcols), ignore_index=True)
rfe2_vars = cols_used(rfe2)
    

#RFE with SMOTE
model4_SMOTE = DecisionTreeClassifier()
rfe2_SMOTE = RFECV(model4_SMOTE, n_jobs=10)
rfe2_SMOTE.fit(X_train_SMOTE,y_train_SMOTE)
y_pred4_SMOTE=rfe2_SMOTE.predict(X_test)

resultsdf=resultsdf.append(pd.Series(graph_metrics(y_pred4_SMOTE,'Tree RFE SMOTE', rfe2_SMOTE), index= resultcols), ignore_index=True)
rfe2_SMOTE_vars = cols_used(rfe2_SMOTE)


print(resultsdf)

resultsdf.to_csv('modelresults.csv') 

print("--- %s seconds ---" % (time.time() - start_time))




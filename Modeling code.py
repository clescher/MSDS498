# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:49:55 2020

@author: clesc
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#may need below line to get imblearn to work
#conda install -c conda-forge imbalanced-learn

os.chdir("C:/Users/clesc/OneDrive/Documents/Northwestern/MSDS 498")
df =  pd.read_csv('modelingdftrain.csv', sep=','  , engine='python')

df = df.drop('Unnamed: 0', axis=1)

#this is the full modeling file once we fix the data issues
X = df[['id', 'default_ind', 'loan_amnt', 'term', 'int_rate', 'grade', 
              'emp_length2', 'home_ownership2', 'annual_inc', 
              'pymnt_plan2', 'desc2', 'purpose', 'dti', 'delinq_2yrs', 
              'revol_util', 'initial_list_status2', 'application_type', 
              'tot_coll_amt', 'chargeoff_within_12_mths', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 
              'total_bal_ex_mort', 'hardship_flag2', 'delinq_amt_pct', 
              'sats_pct', 'max_fico_low','len_credit' ]]

#this is the dataframe that works with current data issues
X = df[['loan_amnt', 'purpose', 'term', 'application_type', 'grade', 'int_rate',
        'emp_length2', 'home_ownership2',
        'pymnt_plan2', 'desc2', 
        'initial_list_status2',  
        'hardship_flag2', 
        'max_fico_low','len_credit']]

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


#declares logistic regresion
logistic_regression= LogisticRegression(max_iter=10000)
model1 = logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

#confusion matrix that's been normalized
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'],normalize=True)
sns.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()

#calculates and plots the ROC and auc
y_pred_proba = logistic_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

##################
#####
###
#test ground for SMOTE
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=13)

columns = X_train.columns
os_data_X,os_data_y = sm.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['default_ind'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['default_ind']==0]))
print("Number of subscription",len(os_data_y[os_data_y['default_ind']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['default_ind']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['default_ind']==1])/len(os_data_X))

#this allows you to use os_data_X and os_data_y as your oversampled training sets
logistic_regression= LogisticRegression(max_iter=10000)
model2 = logistic_regression.fit(os_data_X,os_data_y)
y_pred=logistic_regression.predict(X_test)

#confusion matrix that's been normalized
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'],normalize=True)
sns.heatmap(confusion_matrix, annot=True)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()

#calculates and plots the ROC and auc
y_pred_proba = logistic_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

##################
#####
###
#test ground recursive feature elimination 

data_final_vars=X.columns.values.tolist()

from sklearn.feature_selection import RFE

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

import statsmodels.api as sm
logit_model=sm.Logit(os_data_y,os_data_X)
result=logit_model.fit()
print(result.summary2())



##################
#####
###
#test ground for retreiving summary stats
import sklearn.metrics as metrics
def summary(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))




















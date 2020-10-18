# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:54:18 2020

@author: clesc
"""


# Initiate Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import os

# Import CSV
os.chdir("C:/Users/clesc/OneDrive/Documents/Northwestern/MSDS 498")
df =  pd.read_csv('smallerdata.csv', sep=','  , engine='python')

#Declare all Functions here
#a function that receives the dataframe, data column & row you wish to alter a string into a date format
def dateconvert(datestring):
    #empty 
    if type(datestring) == float:
        year = "1900"
        month = "Jan"
    #3-Jan for 01/01/2003
    elif len(datestring)==5:
        year = "200" + datestring[0:1]
        month = datestring[2:5]
    #19-Mar for 03/01/2019
    elif datestring[0:2].isnumeric():
        year = datestring[0:2]
        month = datestring[3:6]
    #Feb-2000 for 02/01/2000    
    elif datestring[4:8].isnumeric() and len (datestring[4:8])==4:
        month = datestring[0:3]
        year = datestring[4:8]
    #Feb-01 for 02/01/2019
    elif datestring[4:8].isnumeric() and len (datestring[4:8])==2:
        month = datestring[0:3]
        year = datestring[4:6]
    
    #this is a manual process to convert 2 year dates to 4 since the automatic one doesn't work
    if len(year)==2:
        if int(year) < 21:
            year="20"+year
        else:
            year = "19"+year
    
    date_time_str = month +' 01 '+ year
    #all dates have to be forced to a 4 year, otherwise we get like 2065 as dates
    date_time_obj = datetime.datetime.strptime(date_time_str, '%b %d %Y')
    
    return date_time_obj
#Data Cleaning Code
#id
#removes those wierd summary rows as they all contain the word "amount" in the id column
#converts ID to string to then search it, then changes it back to integer to ensure no issues
df['id']=df['id'].astype(str)
df = df[~df['id'].str.contains("amount")]
df['id']=df['id'].astype(np.int32)
#resets the index in order so further coding can be easier
df = df.reset_index(drop=True)


#Date related columns
#add the column headers you need to this date converting list
datecol_list = ['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d',
                'hardship_start_date','hardship_end_date','payment_plan_start_date',
                'debt_settlement_flag_date','settlement_date']

#loops the date columns through the dateconvert function and creates new columns with a "2" at the end
for c in range(0,len(datecol_list)):
    df[datecol_list[c]+"2"] = df.apply(lambda x: dateconvert(x[datecol_list[c]]), axis =1)
    df[datecol_list[c]+"2"] = pd.to_datetime(df[datecol_list[c]+"2"])

#int_rate
# convert 'int_rate' to string
df['int_rate']=df['int_rate'].astype(str)
# strip off % sign and convert to float
df['int_rate'] = df['int_rate'].str.rstrip('%').astype('float') / 100.0

#revol_util
# convert 'revol_util' to string
df['revol_util']=df['revol_util'].astype(str)
# strip off % sign and convert to float
df['revol_util'] = df['revol_util'].str.rstrip('%').astype('float') / 100.0



#Feature Generation
#len_credit
#calculates the length of credit they've had in years
df['len_credit'] = pd.to_numeric((df['issue_d2']-df['earliest_cr_line2']).dt.days)/365


#max_fico_high
df['max_fico_high']= df[["fico_range_high", "sec_app_fico_range_high"]].max(axis=1)

#max_fico_low
df['max_fico_low']= df[["fico_range_low", "sec_app_fico_range_low"]].max(axis=1)

#delinq_amt_pct 
df['delinq_amt_pct']=(df['delinq_amnt']/df['total_bal_ex_mort'])
df.loc[df['delinq_amt_pct']> 1, 'delinq_amt_pct'] = 1

#sats_pct 
df['sats_pct']=(df['num_sats']/df['open_acc'])

#emp_length to numeric
df['emp_length2'] = df['emp_length'].str[0:1]
df.loc[df['emp_length']=="< 1 year", 'emp_length2'] = "0"
df.loc[df['emp_length']=="10+ years", 'emp_length2'] = ">"
df.loc[df['emp_length2']==">", 'emp_length2'] = "10"
df.loc[df['emp_length'].isnull(), 'emp_length2'] = "0"
df['emp_length2']=df['emp_length2'].astype(int)

#initial_list_status (create dummy)
df.loc[df['initial_list_status']=="w", 'initial_list_status2'] = 0
df.loc[df['initial_list_status']=="f", 'initial_list_status2'] = 1

#hardship_flag (1,0)
df.loc[df['hardship_flag']=="N", 'hardship_flag2'] = 0
df.loc[df['hardship_flag']=="Y", 'hardship_flag2'] = 1

#home_ownership
df['home_ownership2'] = 0
df.loc[df['home_ownership']=="RENT", 'home_ownership2'] = 1
df.loc[df['home_ownership']=="OWN", 'home_ownership2'] = 2
df.loc[df['home_ownership']=="MORTGAGE", 'home_ownership2'] = 2


#desc
df['desc2'] = 1
df.loc[df['desc'].isnull(), 'desc2'] = 0

#verification_status
df.loc[df['verification_status']=="Not Verified", 'verification_status2'] = 0
df.loc[df['verification_status']=="Verified", 'verification_status2'] = 1
df.loc[df['verification_status']=="Source Verified", 'verification_status2'] = 1

#pymnt_plan
df.loc[df['pymnt_plan']=="n", 'pymnt_plan2'] = 0
df.loc[df['pymnt_plan']=="y", 'pymnt_plan2'] = 1


#default_ind
#creates binary indicator for defaulted or not
#just an error checker to ensure that all statuses are accounted for
df['default_ind']=2
#should be a lsit of all good status loans
df.loc[(df['loan_status'] == 'Fully Paid'), 'default_ind'] = 0
#should be a list of all defaulted loans
df.loc[(df['loan_status'] == 'Charged Off'), 'default_ind'] = 1

#Loan Amnt Cube Root Transform
df['loan_amnt_cuberoot']=np.power(np.sign(df['loan_amnt']) * np.abs(df['loan_amnt']),1/3)

#Int Rate Log Transform
df['int_rate_log']=np.log10(df['int_rate'])

#Len Credit Cube Root Transform
df['len_credit_cuberoot']=np.power(np.sign(df['len_credit']) * np.abs(df['len_credit']),1/3)



#Row/bad data deletion

df['dti'].replace('', np.nan, inplace=True)
df['revol_util'].replace('', np.nan, inplace=True)
df['tot_coll_amt'].replace('', np.nan, inplace=True)
df['delinq_amt_pct'].replace('', np.nan, inplace=True)
df['pct_tl_nvr_dlq'].replace('', np.nan, inplace=True)
df['pub_rec_bankruptcies'].replace('', np.nan, inplace=True)

#Bad dates
#removes rows with empty dates as they were all set to 01/01/1990
df=df[(df['issue_d2']!="1900-01-01") | (df['earliest_cr_line2']!="1900-01-01")]

df = df.reset_index(drop=True)

#Modeling data set creation
#keep only columns we need
modeldf = df[['id', 'default_ind', 'loan_amnt_cuberoot', 'term', 'int_rate_log', 'grade', 
              'emp_length2', 'home_ownership2', 'annual_inc', 
              'desc2', 'purpose', 'dti', 'delinq_2yrs', 
              'revol_util', 'initial_list_status2', 'application_type', 
              'tot_coll_amt', 'chargeoff_within_12_mths', 'pct_tl_nvr_dlq', 'pub_rec_bankruptcies', 
              'total_bal_ex_mort', 'delinq_amt_pct', 
              'sats_pct', 'max_fico_low','len_credit_cuberoot' ]]


modeldf=modeldf.dropna()

modeldf = modeldf.reset_index(drop=True)

#split into training and testing 
#needs to actually be coded in
modeldftrain=modeldf
#save files
modeldftrain.to_csv('modelingdftrain.csv')  

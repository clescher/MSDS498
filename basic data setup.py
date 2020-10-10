# -*- coding: utf-8 -*-
"""
This is our data cleaning file, it will take in the original .csv and 
output a smaller cleaned version of it for modeling
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import numpy as np
import seaborn as sns


########################
########################
########################
##Place all data cleaning functions here
########################
########################
########################
########################
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

################################################################################
#changes working directory and selects file
os.chdir("C:/Users/clesc/OneDrive/Documents/Northwestern/MSDS 498")
df =  pd.read_csv('smallerdata.csv', sep=','  , engine='python')


########################
########################
########################
##Place all data cleaning codes here
########################
########################
########################
########################

#id
#removes those wierd summary rows as they all contain the word "amount" in the id column
#converts ID to string to then search it, then changes it back to integer to ensure no issues
df['id']=df['id'].astype(str)
df = df[~df['id'].str.contains("amount")]
df['id']=df['id'].astype(np.int32)
#resets the index in order so further coding can be easier
df = df.reset_index(drop=True)

#date related columns
#####add the column headers you need to this date converting list
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


########################
########################
########################
##Place all feature generations here
########################
########################
########################
########################

#len_credit
#calculates the length of credit they've had in years
df['len_credit'] = pd.to_numeric((df['issue_d2']-df['earliest_cr_line2']).dt.days)/365

#fico_avg
#calculates their average FICO at loan originiation
df['fico_avg']=(df['fico_range_low']+df['fico_range_high'])/2

#default_ind
#creates binary indicator for defaulted or not
#just an error checker to ensure that all statuses are accounted for
df['default_ind']=2
#should be a lsit of all good status loans
df.loc[(df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Current'), 'default_ind'] = 0
#should be a list of all defaulted loans
df.loc[(df['loan_status'] == 'Charged Off') | (df['loan_status'] == 'Default'), 'default_ind'] = 1



########################
########################
########################
##Place all row deletions here
########################
########################
########################
########################

#removes rows with bad dates
df=df[(df['issue_d2']!="1900-01-01") | (df['earliest_cr_line2']!="1900-01-01")]



########################
########################
########################
##Extract needed columns and save file here
########################
########################
########################
########################

modeldf = df[['default_ind','loan_amnt', 'term', 'int_rate', 'grade', 
              'len_credit', 'fico_avg','id']]

#should add line to split into test and train
modeldf.to_csv('modelingdf.csv')  








########################
########################
########################
##Place all error checking code here
########################
########################
########################
########################

########################
########################
########################
##Josh workspace
########################
########################
########################
########################
# This is a quick and dirty function for quick reviews of variables 
# Example cell follows

# Works on numerical. Categoricals will still return number missing and description but will then throw error. 
# NOT A GREAT function but it saved me some cut and paste time

def analyze_var(var):
    print(var.describe())
    print('\n')
    print('number missing:')
    print(sum(var.isnull()))

    sns.distplot(var, color='g', bins=100)
    

# Example - dataframe['variable']

analyze_var(df['settlement_term'])

# Found this on web Good way to review our variables post clean
# https://stackoverflow.com/questions/37366717/pandas-print-column-name-with-missing-values

def missing_zero_values_table(df):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " + str(mz_table.shape[0]) +
              " columns that have missing values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
        return mz_table

missing_zero_values_table(df)

########################
########################
########################
##Craig Workspace
########################
########################
########################
########################
#function used to make sure dates are logical
df3 = df[(df['len_credit']<0)]
df3 = df3[['issue_d2','earliest_cr_line2','len_credit','id', 'issue_d','earliest_cr_line']]
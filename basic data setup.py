# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
import datetime
import numpy as np
#######################################
##
#
#
#
#
#Josh update this function
#a function that receives the dataframe, data column & row you wish to alter a string into a date format
def dateconvert(datestring):

    #empty 
    if type(datestring) == float:
        year = "1900"
        month = "Jan"
    #3-Jan for 01/01/2003
    elif len(datestring)==5:
        year = "0" + datestring[0:1]
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
    
    date_time_str = month +' 01 '+ year
    #date format switches partiall through the dataset to not include 4 char year
    if len(year)==2:
        date_time_obj = datetime.datetime.strptime(date_time_str, '%b %d %y')  
    else:
        date_time_obj = datetime.datetime.strptime(date_time_str, '%b %d %Y')
    
    return date_time_obj

#changes working directory and selects file
os.chdir("C:/Users/clesc/OneDrive/Documents/Northwestern/MSDS 498")
df =  pd.read_csv('smallerdata.csv', sep=','  , engine='python')

#removes those wierd summary rows as they all contain the word "amount" in the id column
df = df[~df['id'].str.contains("amount")]
#this converts any blanks in this filed to an NA for an easy drop
df['earliest_cr_line'].replace('', np.nan, inplace=True)
df.dropna(subset=['earliest_cr_line'], inplace=True)


#resets the index in order so further coding can be easier
df = df.reset_index(drop=True)



#####add the column headers you need to this date converting list
datecol_list = ['issue_d','earliest_cr_line','last_pymnt_d','last_credit_pull_d',
                'hardship_start_date','hardship_end_date','payment_plan_start_date',
                'debt_settlement_flag_date','settlement_date']

#####################################
#
#
#
#
#
#Josh update below
#loops the date columns through the dateconvert function and creates new columns with a "2" at the end
for c in range(0,len(datecol_list)):
    df[datecol_list[c]+"2"] = df.apply(lambda x: dateconvert(x[datecol_list[c]]), axis =1)
    df[datecol_list[c]+"2"] = pd.to_datetime(df[datecol_list[c]+"2"])


#
#
#
#
#this is my error testing code
for c in range(0,300000):
    print(c)
    print (df.loc[c]['id'])
    print (df.loc[c]['last_pymnt_d'])
    dateconvert(df.loc[c]['last_pymnt_d'])
    
    

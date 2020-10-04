# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
import datetime
import numpy as np

#a function that receives the dataframe, data column & row you wish to alter a string into a date format
def dateconvert(datestring):
    month = datestring[0:3]
    year = datestring[4:8]
    date_time_str = month +' 01 '+ year
    #date format switches partiall through the dataset to not include 4 char year
    if (len(year)<4):
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
datecol_list = ['issue_d','earliest_cr_line']

#loops the date columns through the dateconvert function and creates new columns with a "2" at the end
for c in range(0,len(datecol_list)):
    df[datecol_list[c]+"2"] = df.apply(lambda x: dateconvert(x[datecol_list[c]]), axis =1)



#
#
#
#
#this is my error testing code
#for c in range(150000,300000):
    #print (df.loc[c]['earliest_cr_line'])
    #dateconvert(df.loc[c]['earliest_cr_line'])
    
    

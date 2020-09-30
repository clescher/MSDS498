

#### Libraries ##########

library(ggplot2)
library(gridExtra)
library(RColorBrewer)
library(kableExtra)
library(lessR)
library(car)
library(psych)
library(plyr)
# library(Hmisc)
library(dplyr)




# Functions for Data Conversion

dateconvert <- function(x){
  
  #grabs the first 3 digits which should be the month
  month<- substr(x, 1, 3)
  #grabs the last 4 digits which should be the year
  year <- substr(x,5,8)
  #pastes it together into a string with the 1st of the month
  thedate <- paste(month,"-01-",year, sep="")
  #returns as a date
  return(as.Date(thedate, format = "%b-%d-%y"))
}



#############################################################################

# Apply date conversion to smallerdata

df1 <- data.frame(smallerdata)

# View(df1)

df1$issue_date <- dateconvert(df1$issue_d)

# Check original issue date with converted date
print(df1[1:3,c(17,153)])


# Alternate  method for date convert

df1$issue_date2 <- as.Date(as.character(df1$issue_d), format = "%b-%Y")

# Check converted date values
print(df1[1:3,c(17,153:154)])








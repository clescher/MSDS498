set.seed(13)

setwd("C:/Users/clesc/OneDrive/Documents/Northwestern/MSDS 498")
data <- read.csv("smallerdata.csv")


#summary(data)

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

data$last_credit_pull_d2 <- dateconvert(data$last_credit_pull_d)

#proof that all NA's aren't my fault rather the data is missing :)
data1 <- data[is.na(data$last_credit_pull_d2),]
data1$last_credit_pull_d

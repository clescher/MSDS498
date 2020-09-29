set.seed(13)

setwd("C:/Users/clesc/OneDrive/Documents/Northwestern/MSDS 498")
data <- read.csv("smallerdata.csv")


#summary(data)

dateconvert <- function(x){

  month<- substr(x, 1, 3)
  year <- substr(x,5,8)
  thedate <- paste(month,"-01-",year, sep="")
  return(as.Date(thedate, format = "%b-%d-%y"))
}

data$last_credit_pull_d2 <- dateconvert(data$last_credit_pull_d)
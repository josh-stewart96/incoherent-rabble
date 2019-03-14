# Load necessary packages
library(dplyr)


##################################################################
## Functions used ################################################
##################################################################

convert_cols_to_factor <- function(data_frame, cols_to_convert) {
  
  data_frame[,fact_cols] <- data_frame %>%
                              select(cols_to_convert) %>%
                              mutate_all(funs(as.factor(.)))
  
  return(data_frame)
}

##################################################################
## Analysis ######################################################
##################################################################

# Read in the raw data
train_df <- tbl_df(read.csv('~/Kaggle/House Prices/train.csv'))
test_df <- tbl_df(read.csv('~/Kaggle/House Prices/test.csv'))

# Identify categorical variables that were read in as numerical, and convert them to factors
fact_cols <- c('MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold')

train_df <- convert_cols_to_factor(train_df, fact_cols)
test_df <- convert_cols_to_factor(test_df, fact_cols)


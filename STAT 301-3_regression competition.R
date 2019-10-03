# STAT 301-3: Data Science III
# Kaggle Competition #2 (Regression)
# John Lee

# Note: this is the replication file for one of my final submissions, but my original R script is much 
# longer (~ 1,000 lines of code). I decided to submit a RF model that uses a transformation of the DV to correct
# for skew (and then the inverse of that transformation to the predicted values, to return to original scale)
# But I also tried dozens of different model specifications including for the OLS, lasso, ridge, and NN as well 


# Loading packages
library(tidyverse)
library(modelr)
library(skimr)
library(readr) # To write and save RDS files
library(janitor)
library(glmnet) # ridge and lasso
library(glmnetUtils) # improves working with glmnet
library(tree) # for CART
library(randomForest) 
library(onehot)
library(missForest) # for prodNA() -- for seeding the MI 
library(xgboost) # for XG boost
library(mlr)
library(tuneRanger) # for tuning RF

set.seed(3)



# Part 1: Load and clean the data ------------------------------------------------------

# Create two versions of both the training and test sets: 
# (v1) std cont predictors; (v2) std cont predictors + OHE cat vars (for the NN)
# note: do this for the test set first -- b/c the var category has 42 unique values in the train set
# but only 39 in the test set --> so I have to drop the obs in the train set with category var values
# that don't appear in the test set; 



# Part 1a: create two versions of the test set  ----------------> v1, v2 ---------------------

# Load the test set
test_dat <- read_csv("data/test.csv") %>%
  mutate_if(is.character, as.factor)

# Inspect the test set 
test_dat %>% skim

# The list of cats that appear in the test set (we'll only use these cats in the training set)
test_cats <- test_dat %>% count(category) %>% pull(category) %>% as.character()

# Extract the id var as a vector
test_id_vec <- test_dat %>% pull(id)

# Test set: only std preds 
test_std <- test_dat %>%
  select(-category, -region, -id) %>% # drop the cat vars + id
  scale() %>% # standardize the continuous predictors (mean = 0, sd = 1)
  as_tibble()

# Create a train set w/ just the cat preds and dv 
test_catsdv <- test_dat %>%
  select(category, region)

# Merge the std cont preds w/ cats + dv to create train set (v1) ---------
test_v1 <- base::cbind(test_std, test_catsdv) %>% as_tibble()

# Inspect test v1 
test_v1 %>% skim


# Create the ohe version of the cat vars 
ohe_test <- test_dat %>% 
  # Only select the cat predictors
  select(category, region) %>%
  onehot::onehot(max_levels = 40) %>% # use onehot to encode variables
  predict(as_tibble(test_dat)) %>% # get OHE matrix
  as_tibble() # Convert back to tibble

# Create the combined test set w/ the ohe cat vars 

test_v2 <- base::cbind(test_v1 %>% select(-category, -region), ohe_test) %>% 
  as_tibble() %>%
  clean_names()




# Part 1b: Create two versions of the train set  ----------------> v1, v2 ----------------------

# Load the train set
train_dat <- read_csv("data/train.csv") %>%
  mutate_if(is.character, as.factor)

# Inspect the data -- there's a lot of missing data; so I'll do lwd 
train_dat %>%
  skim

# Use lwd and just keep the complete cases 
train_dat <- train_dat %>%
  na.omit() %>%
  as_tibble()

# Now, drop the obs with values for the category var that don't appear in the test set 
train_dat <- train_dat %>%
  mutate(exclude_obs = ifelse(category %in% test_cats, 0, 1)) %>%
  # Just filter in the obs with the common cat values (8,878 out of 8,886)
  filter(exclude_obs == 0) %>%
  select(-exclude_obs) %>% # drop the tag 
  mutate(category = droplevels(category)) # drop the factor levels that aren't used

# Verify that there are only 39 values for the category var
train_dat %>% count(category)


# Train set: only std preds 
train_std <- train_dat %>%
  select(-category, -region, -total_funding_usd) %>% # drop the cat vars and the DV
  scale() %>% # standardize the continuous predictors (mean = 0, sd = 1)
  as_tibble()

# Create a train set w/ just the cat preds and dv 
train_catsdv <- train_dat %>%
  select(category, region, total_funding_usd)

# Merge the std cont preds w/ cats + dv to create train set (v1) ---------
train_v1 <- base::cbind(train_std, train_catsdv) %>% as_tibble()


# Create the ohe version of the cat vars 
ohe_train <- train_dat %>% 
  # Only select the cat predictors
  select(category, region) %>%
  onehot::onehot(max_levels = 40) %>% # use onehot to encode variables
  predict(as_tibble(train_dat)) %>% # get OHE matrix
  as_tibble() # Convert back to tibble


# Create the combined test set w/ the ohe cat vars 
train_v2 <- base::cbind(train_v1 %>% select(-category, -region), ohe_train) %>% 
  as_tibble() %>%
  clean_names()


train_v2 %>% skim




# Part 2: CV -----------------------------------------------------------------------

# Part 2a -- first, I'll tune the mtry parameter --------- 

# Set up a tibble with different values for mtry 
model_def <- tibble(mtry = 1:(ncol(train_v1) - 1))

# Returns a random forest where burned is the outcome
fit_rf <- function(data, mtry){
  return(randomForest(total_funding_usd ~ ., 
                      data = data, mtry = mtry, ntree = 100, importance = TRUE))
}

# Perform the 10-fold CV to find the optimal mtry
rf_10fold <- train_v1 %>% 
  crossv_kfold(10, id = "fold") %>%
  # Create 10 folds for each unique value of mtry
  crossing(model_def) %>%
  # Fit the models and compute the fold MSE 
  mutate(model_fit = map2(train, mtry, fit_rf),
         fold_mse = map2_dbl(model_fit, test, modelr::mse))

# Based on the 10-fold CV process, it looks like 4 is the optimal value for mtry. 
# So for my RF candidate model, I will use mtry = 4.

# Display the results 
rf_10fold %>% 
  group_by(mtry) %>% 
  summarize(test_mse = mean(fold_mse)) %>%
  arrange(test_mse)

# Plot the results
rf_10fold %>% 
  group_by(mtry) %>% 
  summarize(test_mse = mean(fold_mse)) %>%
  ggplot(aes(x = mtry, y = test_mse)) +
  geom_line() +
  geom_point() 



# Part 2b -- first, I'll tune the min node size parameter ---------------- 

# Set up a tibble with different values for mtry 
model_def_nsize <- tibble(min_nsize = c(10, 20, 30, 40, 50, 60, 70, 80))

# returns a random forest where burned is the outcome
fit_rf_nsize_1 <- function(data, min_nsize){
  return(randomForest(total_funding_usd ~ ., 
                      data = data, 
                      mtry = 4, # fix the mtry at 4, based on previous CV runs 
                      ntree = 100, 
                      nodesize = min_nsize,
                      importance = TRUE))
}

# Perform the 10-fold CV to find the optimal min node size
rf_10fold4 <- train_v1_impcats %>% 
  crossv_kfold(10, id = "fold") %>%
  # Create 10 folds for each unique value of min node size
  crossing(model_def_nsize) %>%
  # Fit the models and compute the fold MSE 
  mutate(model_fit = map2(train, min_nsize, fit_rf_nsize),
         fold_mse = map2_dbl(model_fit, test, modelr::mse))


# Display the results 
rf_10fold4 %>% 
  group_by(min_nsize) %>% 
  summarize(test_mse = mean(fold_mse)) %>%
  arrange(test_mse)

# Plot the results
rf_10fold4 %>% 
  group_by(min_nsize) %>% 
  summarize(test_mse = mean(fold_mse)) %>%
  ggplot(aes(x = min_nsize, y = test_mse)) +
  geom_line() +
  geom_point() 

# Results: it looks like the best min node size is ~50


# Part 3: Fit the final mod on the full training set, generate predictions ---------------------

# Note: try transforming the DV before the inputs and then doing the inverse of the transformations 
# After the predictions (after some experimentation, I found that this yields a marginal improvement)


# Returns a random forest where burned is the outcome
fit_rf_nsize_stg2 <- function(data, min_nsize){
  return(randomForest(total_funding_usd ~ ., 
                      data = data, 
                      mtry = 4, # fix the mtry at 4, based on previous CV runs 
                      ntree = 100, 
                      nodesize = min_nsize,
                      importance = TRUE))
}


# Transform the original full training set before running the final RF model (I'll un-transform after)
train_v1_trf <- train_v1 %>%
  mutate(total_funding_usd = (total_funding_usd+1)^(1/2)) # first, add one b/c there are some zero outcomes

test_pred_trf <- tibble(train = train_v1_trf %>% list,
                        test = test_v1 %>% list) %>% 
  mutate(mod_fit = map2(.x = train, .y = 50, .f = fit_rf_nsize_stg2), # .y = min node size
         pred_values = map2(.x = mod_fit, .y = test, .f = predict)) %>% 
  select(pred_values) %>% 
  unnest(pred_values) %>%
  mutate(id = test_id_vec,
         total_funding_usd = pred_values^(2)) %>% # add back id's of the test obs with pred non-zero outcomes 
  select(id, total_funding_usd) 

# Export to csv
test_pred_trf

write_csv(test_pred_trf, "final_submission_1.csv")


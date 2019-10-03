# STAT 301-3 
# Kaggle Competition 1: Classification
# R Script for Final Submission #1 
# John Lee

# Selected method: RF with parameters chosen through tuning (i.e., mtry, min node size, cut-offs)

# Loading packages
library(tidyverse)
library(modelr)
library(skimr)
library(readr) # to write and save RDS files
library(janitor)
library(tree) # for CART
library(randomForest) 
library(mlr)
library(tuneRanger) # for tuning RF

# Set the seed so that the results are reproducible
set.seed(3)

# Part 1: Load the data and clean up the data ------------------------------------------


# Part 1a: Loading and cleaning the test data ---------------->

# Load the test
test_dat <- read_csv("data/competition_test.csv") %>%
  mutate_if(is.character, as.factor)

# Inspect the data --
test_dat %>%
  skim

test_dat %>% count(region)
test_dat %>% count(category) 

# Save the ID as a col in a tibble
id_vector <- test_dat %>% 
  select(id) 

# Remove the id var b/c it's not necessary in the analysis
test_dat <- test_dat %>% 
  select(-id)

# The list of cats that appear in the test set (I'll only use these cats in training set)
test_cats <- test_dat %>% 
  count(category) %>% 
  pull(category) %>% 
  as.character()

test_cats


# Part 1b: Loading and cleaning the training data -------------------------------->

# Load the training data
train_dat <- read_csv("data/competition_train.csv") %>%
  mutate_if(is.character, as.factor)

# Inspect the data --
train_dat %>%
  skim

train_dat %>% count(status)

train_dat %>% count(region)

train_dat %>% count(category) 

# Inspect the data --
train_dat %>% 
  na.omit() %>%
  skim

# Create the list-wise deletion (lwd) dataset
lwd_train <- train_dat %>% 
  na.omit()

lwd_train %>% skim

# Delete the obs that have cat values that don't appear in the test set 
lwd_train <- lwd_train %>%
  mutate(exclude_obs = ifelse(category %in% test_cats, 0, 1)) %>%
  # Just filter in the obs with the common cat values 
  filter(exclude_obs == 0) %>%
  select(-exclude_obs) %>%
  mutate(category = droplevels(category)) # drop the unused levels



# Part 2: Use CV to tune a RF model -------------------------------->

# First, create the helper function

# Function to calculate RF error rate (applies for both RF and bagging)
error_rate_rf <- function(model, data){
  # Convert the resample object into a tibble
  as_tibble(data) %>% 
    # Generate the pred class of status
    mutate(pred_status = predict(model, newdata = data, type = "class"), 
           # Create a var that compares the predicted status class to the actual/observed class
           error = pred_status != status) %>% 
    # extract the error vector and compute the rate
    pull(error) %>% 
    mean()
}

# Set up a tibble with different values for mtry 
model_def <- tibble(mtry = 1:(ncol(lwd_train) - 1))

# Returns a random forest where status is the outcome
fitRF_status <- function(data, mtry){
  return(randomForest(formula = as.factor(status) ~ ., data = as_tibble(data), 
                      ntree = 50,
                      mtry = mtry, 
                      replace = FALSE, # sample without replacement (to reduce overfitting)
                      #strata = "status",
                      #sampsize = c(100, 100, 100), # remember 10fold CV => 90% of the full sample
                      cutoff = c(.40, .31, .29), # factor level order: "acquired", "closed",   "ipo" 
                      nodesize = 15, # min. node size (bigger it is, less overfitting)
                      importance = TRUE))
}

# Note to self: don't do the stratified sampling, b/c it reduces accuracy 
table(lwd_train$status)

# Perform the 10-fold CV to find the optimal mtry
rf_status_10fold <- lwd_train %>% 
  crossv_kfold(10, id = "fold") %>%
  # Create 10 folds for each unique value of mtry
  crossing(model_def) %>%
  # Fit the models and compute the fold error rate
  mutate(model_fit = map2(train, mtry, fitRF_status),
         fold_error_rate = map2_dbl(model_fit, test, error_rate_rf),
         importance = map(model_fit, randomForest::importance))

# Display the results (best is about ~73% CV accuracy)
rf_status_10fold %>% 
  group_by(mtry) %>% 
  summarize(error_rate = mean(fold_error_rate)) %>%
  arrange(error_rate) %>%
  skim

# Display the results -- it looks like the ideal mtry is 3 or 4
rf_status_10fold %>% 
  group_by(mtry) %>% 
  summarize(error_rate = mean(fold_error_rate)) %>%
  arrange(error_rate) %>%
  ggplot(aes(x = mtry, y = error_rate)) +
  geom_point() +
  geom_line() +
  geom_smooth()

# This is to check the confusion matrices - so I know where the misclassification is concentrated
rf_status_10fold %>% 
  filter(fold_error_rate > 0.18, mtry == 3) %>% 
  select(model_fit) %>% 
  pull(model_fit)

rf_status_10fold

# Custom function that adds the row name (var) as the var to the importance object
add_imp_vars <- function(matrix_obj){
  
  matrix_obj <- as.data.frame(matrix_obj)
  names_vec <- row.names(matrix_obj)
  
  # Add the vec of var names then save as a tibble
  matrix_obj$names_vec = names_vec
  as_tibble(matrix_obj) # Then convert df to a tibble
}

# Compute the avg var importance across the estimates (from 10-fold CV)
rf_status_10fold %>%
  # Run the function that adds the var name to the impportance matrix
  mutate(imp_vars_tbl = map(importance, add_imp_vars)) %>%
  # Just filter in the imp vars tibbles, then unnest (combine) the tibbles
  select(imp_vars_tbl) %>%
  unnest(imp_vars_tbl) %>%
  # Group by var name, compute the avg importance metric for each var
  group_by(names_vec) %>%
  summarize(mean_MeanDecAcc = mean(MeanDecreaseAccuracy)) %>%
  # Arrange vars in order of their importance
  arrange(desc(mean_MeanDecAcc)) 

# Next, find the ideal minimum node size using Tune Ranger (a package I found online)

# Try tuning w tuneranger 
status.task = makeClassifTask(data = lwd_train, target = "status")

estimateTimeTuneRanger(status.task)

res <- tuneRanger(status.task, measure = list(multiclass.brier), num.trees = 40,
                 tune.parameters = c("mtry", "min.node.size"), iters = 70)

res

# Part 3: Fit the final model on the full training set and predict classes on the test set ------------

# First, create a few more helper functions

# Function to generate the pred class for the test set 
pred_class <- function(model, data){
  # Make sure the data object is a tibble 
  as_tibble(data) %>% 
    # Generate the pred class of status
    mutate(pred_status = predict(model, newdata = data, type = "class"))
}

lwd_train %>% skim
test_dat %>% skim

# Set up the final work flow tibble
final_predict <- tibble(train = lwd_train %>% list(),
                        test = test_dat %>% list(),
                        mtry = 3) %>% # for final submission 1, do mtry = 3; for final submission 2, do mtry = 4
  mutate(model_fit = map2(train, mtry, fitRF_status), # fit the mod using the full lwd train set
         test_pred = map2(model_fit, test, pred_class)) # gen pred status using the test set

# Create a new tibble with just the id vec and the pred class vec 
submission <- final_predict %>%
  select(test_pred) %>%
  unnest(test_pred) %>%
  mutate(id = id_vector %>% pull(id)) %>%
  select(id, pred_status) %>%
  mutate(ID = id, status = pred_status) %>%
  select(ID, status)

submission %>% count(status)

# final submission 1
write_csv(submission, "jlee_final_submission1.csv")





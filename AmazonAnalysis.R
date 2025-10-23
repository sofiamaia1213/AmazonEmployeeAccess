library(tidyverse)
library(patchwork)
library(ggplot2)
library(tidymodels)
library(vroom)
library(GGally)
library(rpart)
library(glmnet)
library(bonsai)
library(lightgbm)
library(agua)
library(h2o)
library(kknn)
library(discrim)
library(naivebayes)

# Read in the data
amazon_train <- vroom("GitHub/AmazonEmployeeAccess/train.csv") %>%
  mutate(ACTION = as.factor(ACTION))
testData <- vroom("GitHub/AmazonEmployeeAccess/test.csv")

# Recipe for preprocessing
amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) 

# Define tunable Naive Bayes model
nb_model <- naive_Bayes(
  Laplace = tune(),
  smoothness = tune()
) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# Workflow
nb_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(nb_model)

# Tuning Grid
nb_grid <- grid_regular(
  Laplace(range = c(0.01, 3)),
  smoothness(range = c(0.01, 3)),
  levels = 5
)

# Cross-validation folds
folds <- vfold_cv(amazon_train, v = 5, strata = ACTION)

# Tune model
nb_tuned <- tune_grid(
  nb_wf,
  resamples = folds,
  grid = nb_grid,
  metrics = metric_set(roc_auc)
)

# Select best hyperparameters
best_nb <- nb_tuned %>%
  select_best(metric = "roc_auc")

# Finalize and fit model on full training data
final_nb_wf <- nb_wf %>%
  finalize_workflow(best_nb) %>%
  fit(amazon_train)

# Predict on test data
predictions <- predict(final_nb_wf, new_data = testData, type = "prob")

# # knn model
# knn_model <- nearest_neighbor(neighbors=5) %>% 
#   set_mode("classification") %>%
# set_engine("kknn")
# 
# knn_wf <- workflow() %>%
# add_recipe(amazon_recipe) %>%
# add_model(knn_model) %>% 
#   fit(data = amazon_train)
# 
# ## Fit or Tune Model HERE
# predictions<- predict(knn_wf, new_data=testData, type="prob")


# # Logistic regression model
# logRegModel <- logistic_reg() %>%
#   set_engine("glm")
# 
# # Workflow
# logReg_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel) %>%
#   fit(data = amazon_train)
# 
# # Make predictions
# amazon_predict <- predict(logReg_wf, new_data = testData, type = "prob")


# my_mod <- rand_forest(mtry = tune(),
#                       min_n=tune(),
#                       trees=500) %>%
# set_engine("ranger") %>%
# set_mode("classification")
# 
# # Workflow
# amazon_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(my_mod)
# 
# # Tuning grid
# param_grid <- parameters(
#   finalize(mtry(), amazon_train),
#   min_n()
# )
# 
# tuning_grid <- grid_regular(param_grid, levels = 8)
# 
# 
# # Cross-validation
# folds <- vfold_cv(amazon_train, v = 2)
# 
# # Tune
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# # Select best
# bestTune <- CV_results %>%
#   select_best(metric = "roc_auc")
# 
# # Finalize workflow and fit
# final_wf <- amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data = amazon_train)
# 
# # Predict
# predictions <- predict(final_wf, new_data = testData, type = "prob")

# Prepare submission
kaggle_submission <- bind_cols(testData, predictions) %>%
  select(id, .pred_1) %>%
  rename(Action = .pred_1)

# Write to CSV
vroom_write(kaggle_submission, file = "GitHub/AmazonEmployeeAccess/NaiveBayes.csv", delim = ",")
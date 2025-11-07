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
library(kernlab)
library(naivebayes)
library(themis)
library(beepr)

# Call h2o
Sys.setenv(JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-25.0.0.36-hotspot")
h2o::h2o.init()

# --- Read Data ---
amazon_train <- vroom("GitHub/AmazonEmployeeAccess/train.csv") %>%
  mutate(ACTION = as.factor(ACTION))
testData <- vroom("GitHub/AmazonEmployeeAccess/test.csv")

# --- Recipe ---
amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# --- Define H2O AutoML model ---
auto_model <- auto_ml() %>%
  set_engine("h2o",
             max_runtime_secs = 1800,
             max_models = 50,
             seed = 17,
             stopping_metric = "AUC") %>%
  set_mode("classification")

# --- Workflow ---
automl_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(auto_model)

# --- Fit model ---
final_fit <- automl_wf %>%
  fit(data = amazon_train)

# --- Predict on test data ---
predictions <- predict(final_fit, new_data = testData, type = "prob")

# --- Prepare submission ---
kaggle_submission <- bind_cols(testData, predictions) %>%
  select(id, .pred_p1) %>%
  rename(Action = .pred_p1)

vroom_write(kaggle_submission,
            file = "GitHub/AmazonEmployeeAccess/H2O_AutoML.csv",
            delim = ",")

# # SVM Recipe
# amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
#   step_zv(all_predictors()) %>%                   # Remove zero-variance predictors
#   step_normalize(all_numeric_predictors()) %>%    # Scale numeric predictors
#   step_pca(all_numeric_predictors(), threshold = 0.99) %>%  # PCA on numeric predictors
#   step_downsample(ACTION)                         # Balance classes if needed


# rbf_mod<- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")

# poly_mod<- svm_poly(degree = 1, cost = 0.0131) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")

# Define SVM model spec
# svm_linear_model <- svm_linear(cost = 0.0131) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")

# Workflow
# linear_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(svm_linear_model) %>%
#   fit(data = amazon_train)

# rbf_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(rbf_mod) %>%
#   fit(data = amazon_train)

# poly_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(poly_mod) %>%
#   fit(data = amazon_train)


# # Tuning Grid
# poly_grid <- grid_regular(
#   degree(range = c(1, 5)),
#   cost(range = c(1, 5)),
#   levels = 5
# )
# 
# # Cross-validation folds
# folds <- vfold_cv(amazon_train, v = 5, strata = ACTION)
# 
# # Tune model
# poly_tuned <- tune_grid(
#   poly_wf,
#   resamples = folds,
#   grid = poly_grid,
#   metrics = metric_set(roc_auc)
# )
# 
# # Select best hyperparameters
# best_poly <- poly_tuned %>%
#   select_best(metric = "roc_auc")
# 
# # Finalize and fit model on full training data
# linear_svm_wf <- _wf %>%
#   finalize_workflow(best_poly)
# 
# # svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
# #   set_mode("classification") %>%
# # set_engine("kernlab")
# # 
# # svmLinear <- svm_linear(cost=tune()) %>% # set or tune
# #   set_mode("classification") %>%
# # set_engine("kernlab")
# 
# predictions<- predict(poly_wf, new_data=testData, type="prob")

# # Define tuneable Naive Bayes model
# nb_model <- naive_Bayes(
#   Laplace = tune(),
#   smoothness = tune()
# ) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# # Workflow
# nb_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(nb_model)
# 
# # Tuning Grid
# nb_grid <- grid_regular(
#   Laplace(range = c(0.01, 3)),
#   smoothness(range = c(0.01, 3)),
#   levels = 5
# )
# 
# # Cross-validation folds
# folds <- vfold_cv(amazon_train, v = 5, strata = ACTION)
# 
# # Tune model
# nb_tuned <- tune_grid(
#   nb_wf,
#   resamples = folds,
#   grid = nb_grid,
#   metrics = metric_set(roc_auc)
# )
# 
# # Select best hyperparameters
# best_nb <- nb_tuned %>%
#   select_best(metric = "roc_auc")
# 
# # Finalize and fit model on full training data
# final_nb_wf <- nb_wf %>%
#   finalize_workflow(best_nb) %>%
#   fit(amazon_train)
# 
# # Predict on test data
# predictions <- predict(final_nb_wf, new_data = testData, type = "prob")

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
# predictions <- predict(logReg_wf, new_data = testData, type = "prob")


# my_mod <- rand_forest(mtry = tune(),
#                       min_n=tune(),
#                       trees=500) %>%
# set_engine("ranger") %>%
# set_mode("classification")
# 




## ---- eval = FALSE----------------------------------------------------------------------------------------
## install.packages("DoubleML")


## ---- eval = FALSE----------------------------------------------------------------------------------------
## remotes::install_github("DoubleML/doubleml-for-r")


## ---- message = FALSE, warning = FALSE--------------------------------------------------------------------
library(DoubleML)


## ---- eval = TRUE-----------------------------------------------------------------------------------------
library(DoubleML)
alpha = 0.5
n_obs = 500
n_vars = 20
set.seed(1234)
data_plr = make_plr_CCDDHNR2018(alpha = alpha, n_obs = n_obs, dim_x = n_vars,
  return_type = "data.table")



## ---- eval = TRUE-----------------------------------------------------------------------------------------
obj_dml_data = DoubleMLData$new(data_plr, y_col = "y", d_cols = "d")


## ---- eval = TRUE-----------------------------------------------------------------------------------------
# Load mlr3 and mlr3learners package and suppress output during estimation
library(mlr3)
library(mlr3learners)
lgr::get_logger("mlr3")$set_threshold("warn")

# Initialize a random forests learner with specified parameters
ml_l = lrn("regr.ranger", num.trees = 100, mtry = n_vars, min.node.size = 2,
  max.depth = 5)
ml_m = lrn("regr.ranger", num.trees = 100, mtry = n_vars, min.node.size = 2,
  max.depth = 5)
ml_g = lrn("regr.ranger", num.trees = 100, mtry = n_vars, min.node.size = 2,
  max.depth = 5)

doubleml_plr = DoubleMLPLR$new(obj_dml_data,
  ml_l, ml_m, ml_g,
  n_folds = 2,
  score = "IV-type")


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
doubleml_plr$fit()
doubleml_plr$summary()


## ---- eval = TRUE, results = 'hide'-----------------------------------------------------------------------
library(DoubleML)
# Load data as data.table
dt_bonus = fetch_bonus(return_type = "data.table")

# output suppressed for the sake of brevity
dt_bonus


## ---- eval = TRUE-----------------------------------------------------------------------------------------
obj_dml_data_bonus = DoubleMLData$new(dt_bonus,
  y_col = "inuidur1",
  d_cols = "tg",
  x_cols = c("female", "black", "othrace",
    "dep1", "dep2", "q2", "q3",
    "q4", "q5", "q6", "agelt35",
    "agegt54", "durable", "lusd",
    "husd"))

# Print data backend: Lists main attributes and methods of a DoubleMLData object
obj_dml_data_bonus


## ---------------------------------------------------------------------------------------------------------
# Print data set (output suppressed)
obj_dml_data_bonus$data


## ---- eval = TRUE, message=FALSE--------------------------------------------------------------------------
set.seed(31415) # required for reproducability of sample split
learner_l = lrn("regr.ranger", num.trees = 500,
  min.node.size = 2, max.depth = 5)
learner_m = lrn("regr.ranger", num.trees = 500,
  min.node.size = 2, max.depth = 5)
doubleml_bonus = DoubleMLPLR$new(obj_dml_data_bonus,
  ml_l = learner_l,
  ml_m = learner_m,
  score = "partialling out",
  dml_procedure = "dml1",
  n_folds = 5,
  n_rep = 1)
doubleml_bonus


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
doubleml_bonus$fit()
doubleml_bonus$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------
doubleml_bonus$coef
doubleml_bonus$se


## ---- eval = TRUE-----------------------------------------------------------------------------------------
# Array with dim = c(n_obs, n_rep, n_treat)
# n_obs: number of observations in the data
# n_rep: number of repetitions (sample splitting)
# n_treat: number of treatment variables
doubleml_bonus$psi[1:5, 1, 1]


## ---- eval = TRUE-----------------------------------------------------------------------------------------
doubleml_bonus$psi_a[1:5, 1, 1]
doubleml_bonus$psi_b[1:5, 1, 1]


## ---- eval = TRUE-----------------------------------------------------------------------------------------
doubleml_bonus$confint(level = 0.95)


## ---- eval = TRUE, results = 'hide', message = FALSE------------------------------------------------------
learner_g = lrn("regr.ranger", num.trees = 500,
  min.node.size = 2, max.depth = 5)
# Classifier for propensity score
learner_classif_m = lrn("classif.ranger", num.trees = 500,
  min.node.size = 2, max.depth = 5)

doubleml_irm_bonus = DoubleMLIRM$new(obj_dml_data_bonus,
  ml_g = learner_g,
  ml_m = learner_classif_m,
  score = "ATE",
  dml_procedure = "dml1",
  n_folds = 5,
  n_rep = 1)
# output suppressed
doubleml_irm_bonus


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
doubleml_irm_bonus$fit()
doubleml_irm_bonus$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------
set.seed(3141)
n_obs = 500
n_vars = 100
theta = rep(3, 3)

# generate matrix-like objects and use the corresponding wrapper
X = matrix(stats::rnorm(n_obs * n_vars), nrow = n_obs, ncol = n_vars)
y = X[, 1:3, drop = FALSE] %*% theta + stats::rnorm(n_obs)
df = data.frame(y, X)


## ---- eval = TRUE, results = 'hide'-----------------------------------------------------------------------
doubleml_data = double_ml_data_from_data_frame(df,
  y_col = "y",
  d_cols = c("X1", "X2", "X3",
    "X4", "X5", "X6",
    "X7", "X8", "X9",
    "X10"))
# suppress output
doubleml_data


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
# output messages during fitting are suppressed
ml_l = lrn("regr.cv_glmnet", s = "lambda.min")
ml_m = lrn("regr.cv_glmnet", s = "lambda.min")
doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_l, ml_m)

doubleml_plr$fit()
doubleml_plr$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------
doubleml_plr$bootstrap(method = "normal", n_rep_boot = 1000)


## ---- eval = TRUE-----------------------------------------------------------------------------------------
doubleml_plr$confint(joint = TRUE)


## ---- eval = TRUE-----------------------------------------------------------------------------------------
doubleml_plr$p_adjust(method = "romano-wolf")


## ---- eval = TRUE-----------------------------------------------------------------------------------------
doubleml_plr$p_adjust(method = "bonferroni")


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
# output messages during fitting are suppressed
ml_l = lrn("regr.glmnet")
ml_m = lrn("regr.glmnet")
doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_l, ml_m)


## ---- eval = TRUE-----------------------------------------------------------------------------------------
# Note that variable names are overwritten by wrapper for matrix interface
doubleml_plr$set_ml_nuisance_params("ml_m", "X1",
  param = list("lambda" = 0.1))
doubleml_plr$set_ml_nuisance_params("ml_l", "X1",
  param = list("lambda" = 0.09))
doubleml_plr$set_ml_nuisance_params("ml_m", "X2",
  param = list("lambda" = 0.095))
doubleml_plr$set_ml_nuisance_params("ml_l", "X2",
  param = list("lambda" = 0.085))


## ---------------------------------------------------------------------------------------------------------
# output omitted for the sake of brevity
str(doubleml_plr$params)


## ---- include = FALSE, eval = TRUE------------------------------------------------------------------------
# output omitted for the sake of brevity, available in the Appendix
params_external_tuning = doubleml_plr$params


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
doubleml_plr$fit()
doubleml_plr$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------
# load required packages for tuning
library(paradox)
library(mlr3tuning)
# set logger to omit messages during tuning and fitting
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

set.seed(1234)
ml_l = lrn("regr.glmnet")
ml_m = lrn("regr.glmnet")
doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_l, ml_m)



## ---- eval = TRUE-----------------------------------------------------------------------------------------
par_grids = list(
  "ml_l" = ParamSet$new(list(
    ParamDbl$new("lambda", lower = 0.05, upper = 0.1))),
  "ml_m" = ParamSet$new(list(
    ParamDbl$new("lambda", lower = 0.05, upper = 0.1))))


## ---- eval = TRUE-----------------------------------------------------------------------------------------
# Provide tune settings
tune_settings = list(terminator = trm("evals", n_evals = 100),
  algorithm = tnr("grid_search", resolution = 11),
  rsmp_tune = rsmp("cv", folds = 5),
  measure = list("ml_l" = msr("regr.mse"),
    "ml_m" = msr("regr.mse")))


## ---- eval = TRUE,  results = 'hide', message = FALSE-----------------------------------------------------
# execution might take around 50 seconds
# Tune
doubleml_plr$tune(param_set = par_grids, tune_settings = tune_settings)

# output omitted for the sake of brevity, available in the Appendix

# acces tuning results for target variable "X1"
doubleml_plr$tuning_res$X1

# tuned parameters
str(doubleml_plr$params)


## ---- include = FALSE, eval = TRUE------------------------------------------------------------------------
params_internal_tuning = doubleml_plr$params


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
# estimate model and summary
doubleml_plr$fit()
doubleml_plr$summary()


## ---- message = FALSE, eval = TRUE------------------------------------------------------------------------
# First generate some data, ml learners and a data-backend
learner = lrn("regr.ranger", num.trees = 100, mtry = 20,
  min.node.size = 2, max.depth = 5)
ml_l = learner
ml_m = learner
data = make_plr_CCDDHNR2018(alpha = 0.5, n_obs = 100,
  return_type = "data.table")
doubleml_data = DoubleMLData$new(data,
  y_col = "y",
  d_cols = "d")


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
set.seed(314)
doubleml_plr_internal = DoubleMLPLR$new(doubleml_data, ml_l, ml_m, n_folds = 4)
doubleml_plr_internal$fit()
doubleml_plr_internal$summary()


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
doubleml_plr_external = DoubleMLPLR$new(doubleml_data, ml_l, ml_m,
  draw_sample_splitting = FALSE)

set.seed(314)
# set up a task and cross-validation resampling scheme in mlr3
my_task = Task$new("help task", "regr", data)
my_sampling = rsmp("cv", folds = 4)$instantiate(my_task)

train_ids = lapply(1:4, function(x) my_sampling$train_set(x))
test_ids = lapply(1:4, function(x) my_sampling$test_set(x))
smpls = list(list(train_ids = train_ids, test_ids = test_ids))

# Structure of the specified sampling scheme
str(smpls)

# Fit model
doubleml_plr_external$set_sample_splitting(smpls)
doubleml_plr_external$fit()
doubleml_plr_external$summary()


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
# use score "partialling out"
set.seed(314)
doubleml_plr_partout = DoubleMLPLR$new(doubleml_data, ml_l, ml_m,
  score = "partialling out")
doubleml_plr_partout$fit()
doubleml_plr_partout$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------
# Here:
# y: dependent variable
# d: treatment variable
# l_hat: predicted values from regression of Y on X's
# m_hat: predicted values from regression of D on X's
# g_hat: predicted values from regression of Y - D * theta on X's, can be ignored in this example
# smpls: sample split under consideration, can be ignored in this example
score_manual = function(y, d, l_hat, m_hat, g_hat, smpls) {
  resid_y = y - l_hat
  resid_d = d - m_hat

  psi_a = -1 * resid_d * resid_d
  psi_b = resid_d * resid_y
  psis = list(psi_a = psi_a, psi_b = psi_b)
  return(psis)
}


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
set.seed(314)
doubleml_plr_manual = DoubleMLPLR$new(doubleml_data, ml_l, ml_m,
  score = score_manual)
doubleml_plr_manual$fit()
doubleml_plr_manual$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------
library(DoubleML)
# Load data as data.table
dt_bonus = fetch_bonus(return_type = "data.table")
dt_bonus


## ---- eval = TRUE, results = "hide"-----------------------------------------------------------------------
obj_dml_data_bonus = DoubleMLData$new(dt_bonus,
  y_col = "inuidur1",
  d_cols = "tg",
  x_cols = c("female", "black", "othrace",
    "dep1", "dep2", "q2", "q3",
    "q4", "q5", "q6", "agelt35",
    "agegt54", "durable", "lusd",
    "husd"))
# Print data backend: Lists main attributes and methods of a DoubleMLData object
obj_dml_data_bonus


## ---- eval = TRUE-----------------------------------------------------------------------------------------
# Print data set (output suppressed)
obj_dml_data_bonus$data


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------
learner_classif_m = lrn("classif.ranger", num.trees = 500,
  min.node.size = 2, max.depth = 5)

doubleml_irm_bonus = DoubleMLIRM$new(obj_dml_data_bonus,
  ml_g = learner_g,
  ml_m = learner_classif_m,
  score = "ATE",
  dml_procedure = "dml1",
  n_folds = 5,
  n_rep = 1)
# output suppressed
doubleml_irm_bonus


## ---- eval = TRUE-----------------------------------------------------------------------------------------
doubleml_data = double_ml_data_from_data_frame(df,
  y_col = "y",
  d_cols = c("X1", "X2", "X3",
    "X4", "X5", "X6",
    "X7", "X8", "X9",
    "X10"))

# suppress output
doubleml_data


## ---- include = TRUE, eval = FALSE, echo = TRUE-----------------------------------------------------------
## # Output: parameters after external tuning
##
## # tuned parameters
## str(doubleml_plr$params)


## ---- include = TRUE, eval = TRUE, echo = FALSE-----------------------------------------------------------
str(params_external_tuning)


## ---- include = TRUE, eval = FALSE, echo = TRUE-----------------------------------------------------------
## # Output: parameters after internal tuning
##
## # access tuning results for target variable "X1"
## doubleml_plr$tuning_res$X1


## ---- include = TRUE, eval = TRUE, echo = FALSE-----------------------------------------------------------
# acces tuning results for target variable "X1"
doubleml_plr$tuning_res$X1


## ---- eval = FALSE----------------------------------------------------------------------------------------
## # tuned parameters
## str(doubleml_plr$params)


## ---- include = TRUE, eval = TRUE, echo = FALSE-----------------------------------------------------------
str(params_internal_tuning)

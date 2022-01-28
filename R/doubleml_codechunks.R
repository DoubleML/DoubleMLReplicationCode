#### Code chunks used in vignette

## ---- eval = FALSE----------------------------------------------------------------------------------------------------------------------------------------
## install.packages("DoubleML")


## ---- eval = FALSE----------------------------------------------------------------------------------------------------------------------------------------
## remotes::install_github("DoubleML/doubleml-for-r")


## ---- message = FALSE, warning = FALSE--------------------------------------------------------------------------------------------------------------------
library("DoubleML")


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
library("DoubleML")
alpha = 0.5
n_obs = 500
n_vars = 20
set.seed(1234)
data_plr = make_plr_CCDDHNR2018(alpha = alpha, n_obs = n_obs, dim_x = n_vars,
  return_type = "data.table")


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
obj_dml_data = DoubleMLData$new(data_plr, y_col = "y", d_cols = "d")


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
library("mlr3")
library("mlr3learners")
lgr::get_logger("mlr3")$set_threshold("warn")


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
ml_g = lrn("regr.ranger", num.trees = 100, mtry = n_vars, min.node.size = 2,
  max.depth = 5)
ml_m = lrn("regr.ranger", num.trees = 100, mtry = n_vars, min.node.size = 2,
  max.depth = 5)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_plr = DoubleMLPLR$new(obj_dml_data,
  ml_g, ml_m, n_folds = 2, score = "IV-type")


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
doubleml_plr$fit()
doubleml_plr$summary()


## ---- eval = TRUE, results = 'hide'-----------------------------------------------------------------------------------------------------------------------
library("DoubleML")


## ---- eval = TRUE, results = 'hide'-----------------------------------------------------------------------------------------------------------------------
dt_bonus = fetch_bonus(return_type = "data.table")


## ---- eval = TRUE, results = 'hide'-----------------------------------------------------------------------------------------------------------------------
dt_bonus


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
obj_dml_data_bonus = DoubleMLData$new(dt_bonus,
  y_col = "inuidur1", d_cols = "tg",
  x_cols = c("female", "black", "othrace", "dep1", "dep2", "q2", "q3",
    "q4", "q5", "q6", "agelt35", "agegt54", "durable", "lusd", "husd"))


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
obj_dml_data_bonus


## ---------------------------------------------------------------------------------------------------------------------------------------------------------
obj_dml_data_bonus$data


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
df_bonus = fetch_bonus(return_type = "data.frame")


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
class(df_bonus)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
names(df_bonus)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
f_flex = formula(" ~ -1 + (female + black + othrace + dep1 + q2 + q3 +
  q4 + q5 + q6 + agelt35 + agegt54 + durable + lusd + husd)^2")


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
X_flex = model.matrix(f_flex, data = df_bonus)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
df_bonus_flex = data.frame("inuidur1" = df_bonus$inuidur1, X_flex,
  "tg" = df_bonus$tg)
obj_dml_data_bonus_flex = double_ml_data_from_data_frame(df_bonus_flex,
  y_col = "inuidur1", d_cols = "tg")


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
obj_dml_data_bonus_flex2 = double_ml_data_from_matrix(X = X_flex,
  y = df_bonus$inuidur1, d = df_bonus$tg)


## ---- eval = FALSE, echo = FALSE--------------------------------------------------------------------------------------------------------------------------
## # Include this code chunk to verify that flexible example runs through
## # Code chunk not contained in the paper
##
## # based on data.frame
## set.seed(1234)
## learner_check = lrn("regr.lm")
## obj_dml_flex = DoubleMLPLR$new(obj_dml_data_bonus_flex, ml_m = learner_check,
##   ml_g = learner_check)
## obj_dml_flex$fit()
## obj_dml_flex$summary()
##
## # based on matrix
## set.seed(1234)
## obj_dml_flex2 = DoubleMLPLR$new(obj_dml_data_bonus_flex2,
##   ml_m = learner_check, ml_g = learner_check)
## obj_dml_flex2$fit()
## obj_dml_flex2$summary()


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
set.seed(31415)
learner_g = lrn("regr.ranger", num.trees = 500, min.node.size = 2,
  max.depth = 5)
learner_m = lrn("regr.ranger", num.trees = 500, min.node.size = 2,
  max.depth = 5)
doubleml_bonus = DoubleMLPLR$new(obj_dml_data_bonus, ml_m = learner_m,
  ml_g = learner_g, score = "partialling out", dml_procedure = "dml1",
  n_folds = 5, n_rep = 1)
doubleml_bonus


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
doubleml_bonus$fit()
doubleml_bonus$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_bonus$coef
doubleml_bonus$se


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_bonus$psi[1:5, 1, 1]


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_bonus$psi_a[1:5, 1, 1]
doubleml_bonus$psi_b[1:5, 1, 1]


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_bonus$confint(level = 0.95)


## ---- eval = TRUE, results = 'hide', message = FALSE------------------------------------------------------------------------------------------------------
learner_classif_m = lrn("classif.ranger", num.trees = 500, min.node.size = 2,
  max.depth = 5)
doubleml_irm_bonus = DoubleMLIRM$new(obj_dml_data_bonus,
  ml_m = learner_classif_m, ml_g = learner_g, score = "ATE",
  dml_procedure = "dml1", n_folds = 5, n_rep = 1)


## ---- eval = TRUE, results = 'hide', message = FALSE------------------------------------------------------------------------------------------------------
doubleml_irm_bonus


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
doubleml_irm_bonus$fit()
doubleml_irm_bonus$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
set.seed(3141)
n_obs = 500
n_vars = 100
theta = rep(3, 3)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
X = matrix(stats::rnorm(n_obs * n_vars), nrow = n_obs, ncol = n_vars)
y = X[, 1:3, drop = FALSE] %*% theta  + stats::rnorm(n_obs)
df = data.frame(y, X)


## ---- eval = TRUE, results = 'hide'-----------------------------------------------------------------------------------------------------------------------
doubleml_data = double_ml_data_from_data_frame(df, y_col = "y",
  d_cols = c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"))


## ---- eval = TRUE, results = 'hide'-----------------------------------------------------------------------------------------------------------------------
doubleml_data


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
ml_g = lrn("regr.cv_glmnet", s = "lambda.min")
ml_m  = lrn("regr.cv_glmnet", s = "lambda.min")
doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g, ml_m)
doubleml_plr$fit()
doubleml_plr$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_plr$bootstrap(method = "normal", n_rep_boot = 1000)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_plr$confint(joint = TRUE)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_plr$p_adjust(method = "romano-wolf")


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_plr$p_adjust(method = "bonferroni")


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
ml_g = lrn("regr.glmnet")
ml_m  = lrn("regr.glmnet")
doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g, ml_m)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_plr$set_ml_nuisance_params("ml_m", "X1", param =
  list("lambda" = 0.1))
doubleml_plr$set_ml_nuisance_params("ml_g", "X1", param =
  list("lambda" = 0.09))
doubleml_plr$set_ml_nuisance_params("ml_m", "X2", param =
  list("lambda" = 0.095))
doubleml_plr$set_ml_nuisance_params("ml_g", "X2", param =
  list("lambda" = 0.085))


## ---------------------------------------------------------------------------------------------------------------------------------------------------------
str(doubleml_plr$params)


## ---- include = FALSE, eval = TRUE------------------------------------------------------------------------------------------------------------------------
params_external_tuning = doubleml_plr$params


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
doubleml_plr$fit()
doubleml_plr$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
library("paradox")
library("mlr3tuning")
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

set.seed(1234)
ml_g = lrn("regr.glmnet")
ml_m = lrn("regr.glmnet")
doubleml_plr = DoubleMLPLR$new(doubleml_data, ml_g, ml_m)



## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
par_grids = list(
  "ml_g" = ParamSet$new(list(ParamDbl$new("lambda", lower = 0.05,
    upper = 0.1))),
  "ml_m" =  ParamSet$new(list(ParamDbl$new("lambda", lower = 0.05,
    upper = 0.1))))


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
tune_settings = list(terminator = trm("evals", n_evals = 100),
  algorithm = tnr("grid_search", resolution = 11),
  rsmp_tune = rsmp("cv", folds = 5),
  measure = list("ml_g" = msr("regr.mse"), "ml_m" = msr("regr.mse")))


## ---- eval = TRUE,  results = 'hide', message = FALSE-----------------------------------------------------------------------------------------------------
doubleml_plr$tune(param_set = par_grids, tune_settings = tune_settings)


## ---- eval = TRUE,  results = 'hide', message = FALSE-----------------------------------------------------------------------------------------------------
doubleml_plr$tuning_res$X1


## ---- eval = TRUE,  results = 'hide', message = FALSE-----------------------------------------------------------------------------------------------------
str(doubleml_plr$params)


## ---- include = FALSE, eval = TRUE------------------------------------------------------------------------------------------------------------------------
params_internal_tuning = doubleml_plr$params


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
doubleml_plr$fit()
doubleml_plr$summary()


## ---- message = FALSE, eval = TRUE------------------------------------------------------------------------------------------------------------------------
learner = lrn("regr.ranger", num.trees = 100, mtry = 20, min.node.size = 2,
  max.depth = 5)
ml_g = learner
ml_m = learner
data = make_plr_CCDDHNR2018(alpha = 0.5, n_obs = 100, return_type =
  "data.table")
doubleml_data = DoubleMLData$new(data, y_col = "y", d_cols = "d")


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
set.seed(314)
doubleml_plr_internal = DoubleMLPLR$new(doubleml_data, ml_g, ml_m, n_folds = 4)
doubleml_plr_internal$fit()
doubleml_plr_internal$summary()


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
doubleml_plr_external = DoubleMLPLR$new(doubleml_data, ml_g, ml_m,
  draw_sample_splitting = FALSE)


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
set.seed(314)
my_task = Task$new("help task", "regr", data)
my_sampling = rsmp("cv", folds = 4)$instantiate(my_task)

train_ids = lapply(1:4, function(x) my_sampling$train_set(x))
test_ids = lapply(1:4, function(x) my_sampling$test_set(x))
smpls = list(list(train_ids = train_ids, test_ids = test_ids))


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
str(smpls)


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
doubleml_plr_external$set_sample_splitting(smpls)
doubleml_plr_external$fit()
doubleml_plr_external$summary()


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
set.seed(314)
doubleml_plr_partout = DoubleMLPLR$new(doubleml_data, ml_g, ml_m,
  score = "partialling out")
doubleml_plr_partout$fit()
doubleml_plr_partout$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
score_manual = function(y, d, g_hat, m_hat, smpls) {
  resid_y = y - g_hat
  resid_d = d - m_hat
  psi_a = -1 * resid_d * resid_d
  psi_b = resid_d * resid_y
  psis = list(psi_a = psi_a, psi_b = psi_b)
  return(psis)
}


## ---- eval = TRUE, message = FALSE------------------------------------------------------------------------------------------------------------------------
set.seed(314)
doubleml_plr_manual = DoubleMLPLR$new(doubleml_data, ml_g, ml_m,
  score = score_manual)
doubleml_plr_manual$fit()
doubleml_plr_manual$summary()


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
library("DoubleML")
dt_bonus = fetch_bonus(return_type = "data.table")
dt_bonus


## ---- eval = TRUE, results = "hide"-----------------------------------------------------------------------------------------------------------------------
obj_dml_data_bonus = DoubleMLData$new(dt_bonus,
  y_col = "inuidur1", d_cols = "tg",
  x_cols = c("female", "black", "othrace", "dep1", "dep2", "q2", "q3",
    "q4", "q5", "q6", "agelt35", "agegt54", "durable", "lusd", "husd"))


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
obj_dml_data_bonus


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
obj_dml_data_bonus$data


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
learner_classif_m = lrn("classif.ranger", num.trees = 500, min.node.size = 2,
  max.depth = 5)
doubleml_irm_bonus = DoubleMLIRM$new(obj_dml_data_bonus,
  ml_m = learner_classif_m, ml_g = learner_g, score = "ATE",
  dml_procedure = "dml1", n_folds = 5, n_rep = 1)


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_irm_bonus


## ---- eval = TRUE-----------------------------------------------------------------------------------------------------------------------------------------
doubleml_data = double_ml_data_from_data_frame(df, y_col = "y",
  d_cols = c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"))
doubleml_data


## ---- include = TRUE, eval = FALSE, echo = TRUE-----------------------------------------------------------------------------------------------------------
## str(doubleml_plr$params)


## ---- include = TRUE, eval = TRUE, echo = FALSE-----------------------------------------------------------------------------------------------------------
str(params_external_tuning)


## ---- include = TRUE, eval = TRUE, echo = TRUE------------------------------------------------------------------------------------------------------------
doubleml_plr$tuning_res$X1


## ---- eval = FALSE----------------------------------------------------------------------------------------------------------------------------------------
## str(doubleml_plr$params)


## ---- include = TRUE, eval = TRUE, echo = FALSE-----------------------------------------------------------------------------------------------------------
str(params_internal_tuning)


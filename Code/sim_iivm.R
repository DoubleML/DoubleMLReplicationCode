rm(list=ls())
library(DoubleML)
library(parallel)
library(doParallel)
library(ggplot2)
library(data.table)
library(mlr3)
library(mlr3learners)
library(data.table)
library(doRNG)

# Create a new directory plus subdirectories
dir.create("simresults")
dir.create("simresults/iivm")

save_plot = FALSE

if (save_plot) {
  dir.create("Figures")
}

date = Sys.Date()
learner_name = "regr.cv_glmnet"
learner2_name = "classif.cv_glmnet"
learner = lrn(learner_name, s = "lambda.min", nfolds = 5)
learner2 = lrn(learner2_name, s = "lambda.min", nfolds = 5)
# learner_projection = "regr.lm" # only low-dimensional z
n_folds = 5
n_rep_folds = 1
dml_procedure = "dml2"
score = "LATE"
n_obs = 1000
dim_x = 20
theta = alpha = 0.5

# Number of repetitions
R = 500

set.seed(1235)

ncores <- detectCores() - 1
cl <- makeCluster(ncores)
registerDoParallel(cl)

registerDoRNG(seed = 1235)

start2 <- Sys.time()
start2


res <- foreach (r = seq(R), .packages = c("DoubleML", "data.table", "mlr3", "mlr3learners")) %dopar% {

  data_ml = make_iivm_data(n_obs = n_obs, dim_x = dim_x, theta = alpha, alpha_x = 1, return_type = "DoubleMLData")

  double_mliivm = DoubleMLIIVM$new(data_ml,
                                   n_folds = n_folds,
                                   ml_g = learner,
                                   ml_m = learner2,
                                   ml_r = learner2,
                                   dml_procedure = dml_procedure,
                                   score = score)

  double_mliivm$fit()
  confint = double_mliivm$confint()
  coef = double_mliivm$coef
  se = double_mliivm$se
  cover = as.numeric(confint[1] < alpha & alpha < confint[2])

  results = list("coef" = coef, "se" = se, "confint" = confint, "cover" = cover)
}

duration2 <- Sys.time() - start2
duration2

save(res, file = paste0("simresults/iivm/raw_results_sim_IIVM_", learner_name, "_n_", n_obs, "_p_", dim_x, ".Rda"))

coef = vapply(res, function(x) x$coef, double(1L))
se = vapply(res, function(x) x$se, double(1L))
coef_resc = (coef - alpha)/se
sd = mean(vapply(res, function(x) x$se, double(1L)))
sd2 = sd(coef)

df_iivm = data.table("coef" = coef, "alpha" = alpha, "coef_resc" = coef_resc)

coverage = sum(vapply(res, function(x) x$cover, double(1L)))/R
print(paste("Coverage:", coverage))

g_iivm = ggplot(df_iivm, aes(x = coef_resc)) +
  geom_density(fill = "dark blue", alpha = 0.3) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.1, fill = "dark blue", color = "black") +
  geom_vline(aes(xintercept = 0), col = "red") +
  geom_vline(aes(xintercept = mean(coef_resc)), col = "dark blue", alpha = 0.3) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.1) +
  xlim(c(-5,5)) +  xlab("Coef") + ylab("Density") +
  ggtitle(paste0("n = ", n_obs, ", p = ", dim_x, ", Coverage = ", coverage, ", Learner = ", learner_name, ", ", learner2_name)) +
  theme_minimal() +
  theme(plot.title = element_text(face="bold", hjust = 0.5))

g_iivm

if (save_plot) {
  ggsave(filename = paste0("Figures/densplot_IIVM_", n_obs, "_", dim_x, "_",
                           "_", dml_procedure, "_", n_folds,
                           "_", n_rep_folds, "_", learner_name, "_", R,
                           "_", alpha, ".pdf"),
         plot = g_iivm)
}

results_summary = data.table(
  "learner" = learner_name,
  "n_folds" = n_folds,
  "n_rep_folds" = n_rep_folds,
  "dml_procedure" = dml_procedure,
  "score" = score,
  "n" = n_obs,
  "p_x" = dim_x,
  "p_z" = NULL,
  "alpha_0" = alpha,
  "param" = NULL,
  "R" = R,
  "coverage" = coverage,
  "coef" = list(coef),
  "Date" = date)

fwrite(results_summary, file = paste0("simresults/iivm/results_IIVM_", n_obs, "_", dim_x, "_",
                                      dml_procedure, "_", n_folds,
                                      "_", n_rep_folds, "_", learner_name, "_",
                                      "_", R,
                                      "_", alpha, ".csv"))

stopCluster(cl)

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
dir.create("simresults/plr")

date = Sys.Date()
learner_name = "regr.ranger"

n_folds = 5
n_rep_folds = 1
dml_procedure = "dml2"
score = "partialling out"
n_obs = 500
dim_x = 20
alpha = 1

if (n_obs == 500 & dim_x == 20 & learner_name == "regr.ranger") {
  learner = ml_m = lrn(learner_name, num.trees = 378, max.depth = 3, mtry = 20, min.node.size = 6)
  ml_g = lrn(learner_name , num.trees = 132, max.depth = 5, mtry = 12, min.node.size = 1)
}

if (learner_name == "regr.cv_glmnet") {
  learner = ml_g = ml_m = lrn(learner_name, s = "lambda.min", nfolds = 5)
}

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

  data_ml = make_plr_CCDDHNR2018(n_obs = n_obs, dim_x = dim_x, alpha = alpha, return_type = "DoubleMLData")

  double_mlplr = DoubleMLPLR$new(data_ml,
                                 n_folds = n_folds,
                                 ml_g = ml_g,
                                 ml_m = ml_m,
                                 dml_procedure = dml_procedure,
                                 score = score)

  double_mlplr$fit()
  confint = double_mlplr$confint()
  coef = double_mlplr$coef
  se = double_mlplr$se
  cover = as.numeric(confint[1] < alpha & alpha < confint[2])

  results = list("coef" = coef, "se" = se, "confint" = confint, "cover" = cover)
}

duration2 <- Sys.time() - start2
duration2

save(res, file = paste0("simresults/plr/raw_results_sim_PLR_", learner_name, "_n_", n_obs, "_p_", dim_x, ".Rda"))

coef = vapply(res, function(x) x$coef, double(1L))
se = vapply(res, function(x) x$se, double(1L))
coef_resc = (coef - alpha)/se
sd = mean(vapply(res, function(x) x$se, double(1L)))
sd2 = sd(coef)

df = data.table("coef" = coef, "alpha" = alpha, "coef_resc" = coef_resc)

coverage = sum(vapply(res, function(x) x$cover, double(1L)))/R
print(paste("Coverage:", coverage))

g_est = ggplot(df, aes(x = coef_resc)) +
  geom_density(fill = "dark blue", alpha = 0.3) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.1, fill = "dark blue", color = "black") +
  geom_vline(aes(xintercept = 0), col = "red") +
  geom_vline(aes(xintercept = mean(coef_resc)), col = "dark blue", alpha = 0.3) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.1) +
  xlim(c(-5,5)) +  xlab("Coef") + ylab("Density") +
  ggtitle(paste0("n = ", n_obs, ", p = ", dim_x, ", Coverage = ", coverage, ", Learner = ", learner_name)) +
  theme_minimal() +
  theme(plot.title = element_text(face="bold", hjust = 0.5))

g_est

ggsave(filename = paste0("simresults/plr/densplot_PLR_", n_obs, "_", dim_x, "_",
                         "_", dml_procedure, "_", n_folds,
                         "_", n_rep_folds, "_", learner_name, "_", R,
                         "_", alpha, ".pdf"),
       plot = g_est)

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

fwrite(results_summary, file = paste0("simresults/plr/results_PLR_", n_obs, "_", dim_x, "_",
                                      dml_procedure, "_", n_folds,
                                      "_", n_rep_folds, "_", learner_name, "_",
                                      "_", R,
                                      "_", alpha, ".csv"))

stopCluster(cl)

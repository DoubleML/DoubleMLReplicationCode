rm(list=ls())
library(DoubleML)
library(mvtnorm)
library(foreach)

# for parallelization
library(parallel)
library(doParallel)
library(doRNG)

# Create a new directory plus subdirectories
dir.create("simresults")
dir.create("simresults/siminf")

date <- Sys.Date()
date

sessionInfo()

# DGP
DGP_desc5 = function(n, p, betamax = 4, decay = 0.99, threshold = 0, noisevar = 10, rho = 0.9, ...){

  beta = vector("numeric", length = p)

  for (j in 1:p){
    beta[j]= betamax*(j)^{-decay}
  }

  beta[beta<threshold] = 0

  covar=toeplitz(rho^(0:(p-1)))
  diag(covar) = rep(1,p)
  mu = rep(0,p)

  x=rmvnorm(n=n, mean=mu, sigma=covar)
  e = rnorm(n, sd = sqrt(noisevar))
  y = x%*%beta + e

  colnames(x) = paste0("Var", seq( from = 1, to = p ))
  names(y) = "y"
  return(list(data = data.frame(y,x), beta = beta, covar = covar))
}

R = 500

# correlation for Sigma
# setting: "medium"
rho = 0.5
threshold = 0.75
# significance level for ci
sig_level = 0.1
n = 1000
p = 42

# Parallelized loop
ncores <- detectCores() - 1
print(ncores)
cl <- makeCluster(ncores)
registerDoParallel(cl)

registerDoRNG(seed = 1235)

begin = Sys.time()
begin

res <- foreach (r = seq(R), .packages = c("DoubleML", "data.table", "mlr3", "mlr3learners", "mvtnorm")) %dopar% {
  #generate data
  DGP1 <- DGP_desc5(n,p, betamax = 9, decay = 0.99, threshold = threshold, noisevar = 3, rho = rho)
  y <- DGP1$data$y
  x <- as.matrix(DGP1$data[,-1,drop=F])
  beta <- DGP1$beta
  dt = data.table(y,x)
  xnames = names(dt)[-1]

  dml_data = DoubleMLData$new(data = dt, y_col = "y", d_cols = xnames)

  learner = lrn("regr.cv_glmnet", s="lambda.min", nfolds = 5)
  ml_g = learner$clone()
  ml_m = learner$clone()
  dml_plr = DoubleMLPLR$new(dml_data, ml_g, ml_m)

  dml_plr$fit()
  dml_plr$bootstrap(n_rep_boot = 1000)

  coef = dml_plr$coef
  ci = dml_plr$confint(joint=TRUE, level = 1-sig_level)
  p_rw = dml_plr$p_adjust(method = "RW")[,2]
  p_bf = dml_plr$p_adjust(method = "bonferroni")[,2]
  p_holm = dml_plr$p_adjust(method = "holm")[,2]

  result = list("beta_0" = beta, "coef" = coef, "pvals_RW" = p_rw,
                "pvals_bonf" = p_bf, "pvals_holm" = p_holm, "ci" = ci)
}

end = Sys.time()
time = end - begin
time

# save results
save(res , file = paste0("simresults/siminf/results_simInf_n_", n, "_p_", p, "_R_", R, ".Rda"))

# True H0
TH0 = which(res[[1]]$beta < threshold)
FH0 = which(res[[1]]$beta > 0)

# CI
ci_false_rej = lapply(res, function(x) as.numeric(x$ci[TH0,1]*x$ci[TH0, 2]>0))
ci_FWER = mean(sapply(ci_false_rej, function(x) any(x > 0)))
ci_FWER

avg_corr_ci = mean(sapply(res, function(x) sum(x$ci[FH0,1]*x$ci[FH0, 2]>0)))
avg_corr_ci

# RW
pvals_rw = sapply(res, function(x) x$pvals_RW) # rows: variables, # columns: reps
fwer_rw = mean(sapply(res, function(x) any(x$pvals_RW[TH0] < sig_level)))
fwer_rw

avg_corr_rw = mean(sapply(res, function(x) sum(x$pvals_RW[FH0] < sig_level)))
avg_corr_rw

# Bonf
pvals_bf = sapply(res, function(x) x$pvals_bonf) # rows: variables, # columns: reps
fwer_bf = mean(sapply(res, function(x) any(x$pvals_bonf[TH0] < sig_level)))
fwer_bf

avg_corr_bf = mean(sapply(res, function(x) sum(x$pvals_bonf[FH0] < sig_level)))
avg_corr_bf

# Holm
pvals_holm = sapply(res, function(x) x$pvals_holm) # rows: variables, # columns: reps
fwer_holm = mean(sapply(res, function(x) any(x$pvals_holm[TH0] < sig_level)))
fwer_holm

avg_corr_holm = mean(sapply(res, function(x) sum(x$pvals_holm[FH0] < sig_level)))
avg_corr_holm

FWER = c(ci_FWER, fwer_rw, fwer_bf, fwer_holm)
corr_freq = c(avg_corr_ci, avg_corr_rw, avg_corr_bf, avg_corr_holm)

result_table = rbind(FWER, corr_freq)
colnames(result_table) = c("CI", "RW", "Bonf", "Holm")

print(result_table)

library(xtable)
xtable(result_table)

save(result_table, file = paste0("simresults/siminf/summary_table_simInf_n_", n, "_p_", p, "_R_", R, ".Rda"))

stopCluster(cl)

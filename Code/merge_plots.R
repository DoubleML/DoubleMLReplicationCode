# This file collects the simulation results and generates the plots presented in the paper

# Run this file after successful execution of the files
# - sim_plr.R
# - sim_plivX.R
# - sim_irm.R
# - sim_iivm.R
library(ggplot2)
library(data.table)
library(DoubleML)

# Create new subdirectory
dir.create("Figures")

#load raw results and generate plots
# PLR
n_obs = 500
dim_x = 20
alpha = 1
R = 500
learner_name = "regr.ranger"

res_plr = load("simresults/plr/raw_results_sim_PLR_regr.ranger_n_500_p_20.Rda")

coef = vapply(res, function(x) x$coef, double(1L))
se = vapply(res, function(x) x$se, double(1L))
coef_resc = (coef - alpha)/se
sd = mean(vapply(res, function(x) x$se, double(1L)))
sd2 = sd(coef)

df_plr = data.table("coef" = coef, "alpha" = alpha, "coef_resc" = coef_resc)

coverage = sum(vapply(res, function(x) x$cover, double(1L)))/R
print(paste("Coverage:", coverage))

g_plr = ggplot(df_plr, aes(x = coef_resc)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.1, fill = "dark blue", color = "black") +
  # geom_vline(aes(xintercept = alpha), col = "blue") +
  geom_vline(aes(xintercept = 0), col = "red") +
  geom_vline(aes(xintercept = mean(coef_resc)), col = "dark blue", alpha = 0.3) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  xlim(c(-5,5)) +  xlab("Coef") + ylab("Density") +
  labs(title = "PLR", caption = paste0("n = ", n_obs, ", p = ", dim_x, ", Coverage = ", coverage, ", Learner = ", learner_name)) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 5, hjust = 0.5))

g_plr

#  pliv
n = n_obs = 500
p_x = dim_x = 20
p_z = 1
alpha = 1
# Number of repetitions
R = 500

res_pliv = load("simresults/pliv/raw_results_sim_PLIV_regr.cv_glmnet_n_500_p_20.Rda")
learner_name = "regr.cv_glmnet"

coef = vapply(res, function(x) x$coef, double(1L))
se = vapply(res, function(x) x$se, double(1L))
coef_resc = (coef - alpha)/se
sd = mean(vapply(res, function(x) x$se, double(1L)))
sd2 = sd(coef)
df_pliv = data.table("coef" = coef, "alpha" = alpha, "coef_resc" = coef_resc)

coverage = sum(vapply(res, function(x) x$cover, double(1L)))/R
print(paste("Coverage:", coverage))

g_pliv = ggplot(df_pliv, aes(x = coef_resc)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.1, fill = "dark blue", color = "black") +
  geom_vline(aes(xintercept = 0), col = "red") +
  geom_vline(aes(xintercept = mean(coef_resc)), col = "dark blue", alpha = 0.3) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  xlim(c(-5,5)) +  xlab("Coef") + ylab("Density") +
  labs(title = "PLIV", caption = paste0("n = ", n_obs, ", p = ", dim_x, ", Coverage = ", coverage, ", Learner = ", learner_name)) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 5, hjust = 0.5))

g_pliv

# IRM
n_obs = 1000
dim_x = 20
theta = alpha = 0.5
learner_name = "regr.cv_glmnet"
learner2_name = "classif.cv_glmnet"

# Number of repetitions
R = 500
res_irm = load("simresults/irm/raw_results_sim_IRM_regr.cv_glmnet_n_1000_p_20.Rda")

coef = vapply(res, function(x) x$coef, double(1L))
se = vapply(res, function(x) x$se, double(1L))
coef_resc = (coef - alpha)/se
sd = mean(vapply(res, function(x) x$se, double(1L)))
sd2 = sd(coef)

df_irm = data.table("coef" = coef, "alpha" = alpha, "coef_resc" = coef_resc)

coverage = sum(vapply(res, function(x) x$cover, double(1L)))/R
print(paste("Coverage:", coverage))

g_irm = ggplot(df_irm, aes(x = coef_resc)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.1, fill = "dark blue", color = "black") +
  geom_vline(aes(xintercept = 0), col = "red") +
  geom_vline(aes(xintercept = mean(coef_resc)), col = "dark blue", alpha = 0.3) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  xlim(c(-5,5)) +  xlab("Coef") + ylab("Density") +
  labs(title = "IRM", caption = paste0("n = ", n_obs, ", p = ", dim_x, ", Coverage = ", coverage, ", Learner = cv_glmnet")) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 5, hjust = 0.5))

g_irm

# IIVM
n_obs = 1000
dim_x = 20
theta = alpha = 0.5

# Number of repetitions
R = 500

res_iivm = load("simresults/iivm/raw_results_sim_IIVM_regr.cv_glmnet_n_1000_p_20.Rda")

coef = vapply(res, function(x) x$coef, double(1L))
se = vapply(res, function(x) x$se, double(1L))
coef_resc = (coef - alpha)/se
sd = mean(vapply(res, function(x) x$se, double(1L)))
sd2 = sd(coef)

df_iivm = data.table("coef" = coef, "alpha" = alpha, "coef_resc" = coef_resc)

coverage = sum(vapply(res, function(x) x$cover, double(1L)))/R
print(paste("Coverage:", coverage))

g_iivm = ggplot(df_iivm, aes(x = coef_resc)) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.1, fill = "dark blue", color = "black") +
  geom_vline(aes(xintercept = 0), col = "red") +
  geom_vline(aes(xintercept = mean(coef_resc)), col = "dark blue", alpha = 0.3) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  xlim(c(-5,5)) +  xlab("Coef") + ylab("Density") +
  labs(title = "IIVM", caption = paste0("n = ", n_obs, ", p = ", dim_x, ", Coverage = ", coverage, ", Learner = cv_glmnet")) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 5, hjust = 0.5))

g_iivm

library(cowplot)

g_grid = plot_grid(g_plr, g_pliv, g_irm, g_iivm, nrow = 2)
g_grid

ggsave(filename = "Figures/simulations_doubleml_key_models.pdf",
       plot = g_grid, width=5, height=5, dpi=150)

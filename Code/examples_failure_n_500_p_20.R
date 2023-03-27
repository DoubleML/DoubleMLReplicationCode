# Script for replication of figures in DoubleML package vigette
# Failure of naive inference / Building block of DoubleML
library(DoubleML)
library(ggplot2)
library(mlr3)
library(mlr3learners)
library(data.table)
library(cowplot)
lgr::get_logger("mlr3")$set_threshold("warn")

# Create a new directory plus subdirectories
dir.create("simresults")
dir.create("simresults/examples_fail")


set.seed(1234)
n_rep = 1000
n_obs = 500
n_vars = 20
alpha = 0.5

theta_nonorth = theta_orth_nosplit = theta_dml = theta_dml_split_no_cross = rep(0, n_rep)
se_nonorth = se_orth_nosplit = se_dml = rep(0, n_rep)

data = list()
for (i_rep in seq_len(n_rep)) {
  data[[i_rep]] = make_plr_CCDDHNR2018(alpha=alpha, n_obs=n_obs, dim_x=n_vars,
                                       return_type="data.frame")
}

################
### Non orth ###
################
non_orth_score = function(y, d, l_hat, m_hat, g_hat, smpls) {
  u_hat = y - g_hat
  psi_a = -1*d*d
  psi_b = d*u_hat
  psis = list(psi_a = psi_a, psi_b = psi_b)
  return(psis)
}

set.seed(1111)

# Learners with tuned params (external tuning)
ml_l = lrn("regr.ranger", num.trees = 132, max.depth = 5, mtry = 12, min.node.size = 1)
ml_m = lrn("regr.ranger", num.trees = 378, max.depth = 3, mtry = 20, min.node.size = 6)
ml_g = ml_l$clone()

for (i_rep in seq_len(n_rep)) {
  df = data[[i_rep]]
  obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
  obj_dml_plr_nonorth = DoubleMLPLR$new(obj_dml_data,
                                        ml_l, ml_m, ml_g,
                                        n_folds=2,
                                        score=non_orth_score,
                                        apply_cross_fitting=FALSE)
  obj_dml_plr_nonorth$fit()
  this_theta = obj_dml_plr_nonorth$coef
  this_se = obj_dml_plr_nonorth$se
  theta_nonorth[i_rep] = this_theta
  se_nonorth[i_rep] = this_se
}

df_nonorth_resc = data.frame("theta_nonorth" = (theta_nonorth-alpha)/se_nonorth)

g_nonorth_resc = ggplot(df_nonorth_resc, aes(x = theta_nonorth)) +
  geom_histogram(aes(y=after_stat(density)), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
  geom_vline(aes(xintercept = 0), col = "black") +
  xlim(c(-10, 10)) + xlab("") + ylab("") + theme_minimal() +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  ggtitle(paste0("Non-Orthogonal, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_nonorth_resc

ggsave(filename = paste0("simresults/examples_fail/nonorth_resc_n", n_obs, "_p", n_vars, ".pdf"),
       plot = g_nonorth_resc)

################
### no split ###
################
set.seed(2222)

for (i_rep in seq_len(n_rep)){
  df = data[[i_rep]]
  obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
  obj_dml_plr_orth_nosplit = DoubleMLPLR$new(obj_dml_data,
                                             ml_l, ml_m, ml_g,
                                             n_folds=1,
                                             score='IV-type',
                                             apply_cross_fitting=FALSE)
  obj_dml_plr_orth_nosplit$fit()
  this_theta = obj_dml_plr_orth_nosplit$coef
  this_se = obj_dml_plr_orth_nosplit$se
  theta_orth_nosplit[i_rep] = this_theta
  se_orth_nosplit[i_rep] = this_se
}

df_orth_nosplit_resc = data.frame("theta_orth_nosplit" = (theta_orth_nosplit-alpha)/se_orth_nosplit)

g_orth_nosplit_resc = ggplot(df_orth_nosplit_resc, aes(x = theta_orth_nosplit)) +
  geom_histogram(aes(y=after_stat(density)), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
  geom_vline(aes(xintercept = 0), col = "black") +
  xlim(c(-10, 10)) + xlab("") + ylab("") + theme_minimal() +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  ggtitle(paste0("Full Sample, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_orth_nosplit_resc

ggsave(filename = paste0("simresults/examples_fail/orth_nosplit_resc_n", n_obs, "_p", n_vars, ".pdf"),
       plot = g_orth_nosplit_resc)

###########
### DML ###
###########
set.seed(3333)

for (i_rep in seq_len(n_rep)) {
  df = data[[i_rep]]
  obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
  obj_dml_plr = DoubleMLPLR$new(obj_dml_data,
                                ml_l, ml_m, ml_g,
                                n_folds=2,
                                score='IV-type')
  obj_dml_plr$fit()
  this_theta = obj_dml_plr$coef
  this_se = obj_dml_plr$se
  theta_dml[i_rep] = this_theta
  se_dml[i_rep] = this_se
}

df_dml = data.frame("theta_dml" = theta_dml - alpha)
sd_dml = sd(theta_dml)

g_dml = ggplot(df_dml, aes(x = theta_dml)) +
  geom_histogram(aes(y=after_stat(density)), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
  geom_vline(aes(xintercept = 0), col = "black") +
  xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,9)) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = sd_dml), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  ggtitle(paste0("Orthogonal, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml

ggsave(filename = paste0("simresults/examples_fail/double_ml_n", n_obs, "_p", n_vars, ".pdf"),
       plot = g_dml)


df_dml_resc = data.frame("theta_dml" = (theta_dml - alpha)/se_dml)

g_dml_resc = ggplot(df_dml_resc, aes(x = theta_dml)) +
  geom_histogram(aes(y=after_stat(density)), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
  geom_vline(aes(xintercept = 0), col = "black") +
  xlim(c(-10, 10)) + xlab("") + ylab("") + theme_minimal() +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  ggtitle(paste0("Split Sample, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml_resc

ggsave(filename = paste0("simresults/examples_fail/double_ml_resc_n", n_obs, "_p", n_vars, ".pdf"),
       plot = g_dml_resc)


# with heading "Orthogonal ..."
g_dml_orth_resc = ggplot(df_dml_resc, aes(x = theta_dml)) +
  geom_histogram(aes(y=after_stat(density)), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
  # geom_density(fill = "dark blue", alpha = 0.3, color = "dark blue") +
  geom_vline(aes(xintercept = 0), col = "black") +
  xlim(c(-10, 10)) + xlab("") + ylab("") + theme_minimal() +
  stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  ggtitle(paste0("Orthogonal, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml_orth_resc

ggsave(filename = paste0("simresults/examples_fail/double_ml_orth_resc_n", n_obs, "_p", n_vars, ".pdf"),
       plot = g_dml_resc)


# Combined plots
g_comb1_resc = plot_grid(g_nonorth_resc, g_dml_orth_resc)

ggsave(filename = paste0("simresults/examples_fail/nonorth_doubleml_n", n_obs, "_p", n_vars, "_resc.pdf"),
       plot = g_comb1_resc, width=8, height=3, dpi=150)


g_comb2_resc  = plot_grid(g_orth_nosplit_resc, g_dml_resc)

ggsave(filename = paste0("simresults/examples_fail/nosplit_doubleml_n", n_obs, "_p", n_vars, "_resc.pdf"),
       plot = g_comb2_resc, width=8, height=3, dpi=150)


### Cross-Fitting vs. no cross-fitting (both cases with sample splitting) ###
set.seed(3333)

for (i_rep in seq_len(n_rep)) {
  df = data[[i_rep]]
  obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
  obj_dml_plr = DoubleMLPLR$new(obj_dml_data,
                                ml_l, ml_m, ml_g,
                                n_folds=2,
                                score='IV-type',
                                apply_cross_fitting =  FALSE)
  obj_dml_plr$fit()
  this_theta = obj_dml_plr$coef
  theta_dml_split_no_cross[i_rep] = this_theta
}

df_dml_split_no_cross = data.frame("theta_dml_split_no_cross" = theta_dml_split_no_cross - alpha)
sd_dml_split_no_cross = sd(theta_dml_split_no_cross)

g_dml_split_no_cross = ggplot(df_dml_split_no_cross, aes(x = theta_dml_split_no_cross)) +
  geom_histogram(aes(y=after_stat(density)), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
  geom_vline(aes(xintercept = 0), col = "black") +
  xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,9)) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = sd_dml_split_no_cross), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  ggtitle(paste0("No cross-fitting, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml_split_no_cross

ggsave(filename = paste0("simresults/examples_fail/double_ml_resc_no_cross_n", n_obs, "_p", n_vars, ".pdf"),
       plot = g_dml_split_no_cross)


g_dml = ggplot(df_dml, aes(x = theta_dml)) +
  geom_histogram(aes(y=after_stat(density)), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
  geom_vline(aes(xintercept = 0), col = "black") +
  xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,9)) +
  stat_function(fun = dnorm, args = list(mean = 0, sd = sd_dml), geom = "area", col = "red", fill = "red", alpha = 0.01) +
  ggtitle(paste0("Cross-fitting, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml

g_comb3 = plot_grid(g_dml_split_no_cross, g_dml)

ggsave(filename = paste0("simresults/examples_fail/nocrossfit_doubleml_n", n_obs, "_p", n_vars, ".pdf"),
       plot = g_comb3, width=8, height=3, dpi=150)

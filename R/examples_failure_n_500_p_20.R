# Script for replication of figures in DoubleML package vigette
# Failure of naive inference / Building block of DoubleML
library(DoubleML)
library(ggplot2)

# Create a new directory plus subdirectories
dir.create("simresults")
dir.create("simresults/examples_fail")


set.seed(1234)
n_rep = 1000
n_obs = 500
n_vars = 20
alpha = 0.5

theta_ols = theta_nonorth = theta_orth_nosplit = theta_dml = rep(0, n_rep)

data = list()
for (i_rep in seq_len(n_rep)) {
    data[[i_rep]] = make_plr_CCDDHNR2018(alpha=alpha, n_obs=n_obs, dim_x=n_vars,
                                          return_type="data.frame")
}

###########
### OLS ###
###########
est_ols = function(df) {
    ols = stats::lm(y ~ 1 +., df)
    theta = coef(ols)["d"]
    return(theta)
}

for (i_rep in seq_len(n_rep)) {
  df = data[[i_rep]]
  this_theta = est_ols(df)
  theta_ols[i_rep] = this_theta
}

df_ols = data.frame(theta_ols = theta_ols - alpha)
sd_ols = sd(theta_ols)
g_ols = ggplot(df_ols, aes(x = theta_ols)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,9)) +
            stat_function(fun = dnorm, args = list(mean = 0, sd = sd_ols), geom = "area", col = "red", fill = "red", alpha = 0.01) +
            ggtitle(paste0("OLS, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_ols

ggsave(filename = paste0("simresults/examples_fail/ols_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_ols)

df_ols_resc = data.frame(theta_ols_resc = (theta_ols - alpha)/sd_ols)
g_ols_resc = ggplot(df_ols_resc, aes(x = theta_ols_resc)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            # geom_density(fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-10, 10)) + xlab("") + ylab("") + theme_minimal() + # ylim(c(0,9)) +
            stat_function(fun = dnorm, args = list(mean = 0, sd =1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
            ggtitle(paste0("OLS, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))
g_ols_resc


################
### Non orth ###
################
non_orth_score = function(y, d, g_hat, m_hat, smpls) {
 u_hat = y - g_hat
 psi_a = -1*d*d
 psi_b = d*u_hat
 psis = list(psi_a = psi_a, psi_b = psi_b)
 return(psis)
}

library(mlr3)
library(mlr3learners)
library(data.table)
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(1111)

# Learners with tuned params (external tuning)
ml_m = lrn("regr.ranger", num.trees = 378, max.depth = 3, mtry = 20, min.node.size = 6)
ml_g = lrn("regr.ranger", num.trees = 132, max.depth = 5, mtry = 12, min.node.size = 1)

for (i_rep in seq_len(n_rep)) {
    df = data[[i_rep]]
    obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
    obj_dml_plr_nonorth = DoubleMLPLR$new(obj_dml_data,
                                          ml_g, ml_m,
                                          n_folds=2,
                                          score=non_orth_score,
                                          apply_cross_fitting=FALSE)
    obj_dml_plr_nonorth$fit()
    this_theta = obj_dml_plr_nonorth$coef
    theta_nonorth[i_rep] = this_theta
}

df_nonorth = data.frame(theta_nonorth = theta_nonorth - alpha)
sd_nonorth = sd(theta_nonorth)
g_nonorth = ggplot(df_nonorth, aes(x = theta_nonorth)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,9)) +
            stat_function(fun = dnorm, args = list(mean = 0, sd = sd_nonorth), geom = "area", col = "red", fill = "red", alpha = 0.01) +
             ggtitle(paste0("Non-Orthogonal, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_nonorth

df_nonorth_resc = data.frame("theta_nonorth" = df_nonorth$theta_nonorth/sd_nonorth)

g_nonorth_resc = ggplot(df_nonorth_resc, aes(x = theta_nonorth)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-10, 10)) + xlab("") + ylab("") + theme_minimal() +
            stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
            ggtitle(paste0("Full Sample, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_nonorth_resc

ggsave(filename = paste0("simresults/examples_fail/nonorth_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_nonorth)

ggsave(filename = paste0("simresults/examples_fail/nonorth_resc_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_nonorth_resc)

################
### no split ###
################
library(data.table)
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(2222)

for (i_rep in seq_len(n_rep)){
    df = data[[i_rep]]
    obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
    obj_dml_plr_orth_nosplit = DoubleMLPLR$new(obj_dml_data,
                                           ml_g, ml_m,
                                           n_folds=1,
                                           score="partialling out",
                                           apply_cross_fitting=FALSE)
    obj_dml_plr_orth_nosplit$fit()
    this_theta = obj_dml_plr_orth_nosplit$coef
    theta_orth_nosplit[i_rep] = this_theta
}


df_orth_nosplit = data.frame(theta_orth_nosplit = theta_orth_nosplit - alpha)
sd_orth_nosplit = sd(theta_orth_nosplit)
g_orth_nosplit = ggplot(df_orth_nosplit, aes(x = theta_orth_nosplit)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,12)) +
            stat_function(fun = dnorm, args = list(mean = 0, sd = sd_orth_nosplit), geom = "area", col = "red", fill = "red", alpha = 0.01) +
            ggtitle(paste0("Full Sample, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_orth_nosplit

ggsave(filename = paste0("simresults/examples_fail/orth_nosplit_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_orth_nosplit)


df_orth_nosplit_resc = data.frame("theta_orth_nosplit" = df_orth_nosplit$theta_orth_nosplit/sd_orth_nosplit)
g_orth_nosplit_resc = ggplot(df_orth_nosplit_resc, aes(x = theta_orth_nosplit)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
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
                              ml_g, ml_m,
                              n_folds=2,
                              score="partialling out")
    obj_dml_plr$fit()
    this_theta = obj_dml_plr$coef
    theta_dml[i_rep] = this_theta
}

df_dml = data.frame(theta_dml = theta_dml - alpha)
sd_dml = sd(theta_dml)
g_dml = ggplot(df_dml, aes(x = theta_dml)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,9)) +
            stat_function(fun = dnorm, args = list(mean = 0, sd = sd_dml), geom = "area", col = "red", fill = "red", alpha = 0.01) +
            ggtitle(paste0("Orthogonal, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml

ggsave(filename = paste0("simresults/examples_fail/double_ml_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_dml)


df_dml_resc = data.frame("theta_dml" = df_dml$theta_dml/sd_dml)
g_dml_resc = ggplot(df_dml_resc, aes(x = theta_dml)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-10, 10)) + xlab("") + ylab("") + theme_minimal() +
            stat_function(fun = dnorm, args = list(mean = 0, sd = 1), geom = "area", col = "red", fill = "red", alpha = 0.01) +
            ggtitle(paste0("Split Sample, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml_resc

ggsave(filename = paste0("simresults/examples_fail/double_ml_resc_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_dml_resc)


library(cowplot)
g_comb1  = plot_grid(g_nonorth, g_dml)
ggsave(filename = paste0("simresults/examples_fail/nonorth_doubleml_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_comb1, width=8, height=3, dpi=150)

g_comb2  = plot_grid(g_orth_nosplit, g_dml)
ggsave(filename = paste0("simresults/examples_fail/nosplit_doubleml_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_comb2, width=8, height=3, dpi=150)

g_comb1b  = plot_grid(g_nonorth, g_ols)
ggsave(filename = paste0("simresults/examples_fail/nonorth_ols_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_comb1b, width=8, height=3, dpi=150)

# Rescaled version of g_comb2
g_comb2_resc  = plot_grid(g_orth_nosplit_resc, g_dml_resc)

ggsave(filename = paste0("simresults/examples_fail/nosplit_doubleml_n", n_obs, "_p", n_vars, "_resc.pdf"),
        plot = g_comb2_resc, width=8, height=3, dpi=150)

g_comb1_resc = plot_grid(g_nonorth_resc, g_dml_resc)
ggsave(filename = paste0("simresults/examples_fail/nonorth_doubleml_n", n_obs, "_p", n_vars, "_resc.pdf"),
        plot = g_comb1_resc, width=8, height=3, dpi=150)


### Cross-Fitting vs. no cross-fitting (both cases with sample splitting) ###

###########
### DML ###
###########
set.seed(3333)
theta_dml_split_no_cross = rep(0, n_rep)

for (i_rep in seq_len(n_rep)) {
    df = data[[i_rep]]
    obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
    obj_dml_plr = DoubleMLPLR$new(obj_dml_data,
                              ml_g, ml_m,
                              n_folds=2,
                              score="partialling out",
                              apply_cross_fitting =  FALSE)
    obj_dml_plr$fit()
    this_theta = obj_dml_plr$coef
    theta_dml_split_no_cross[i_rep] = this_theta
}

df_dml_split_no_cross = data.frame(theta_dml_split_no_cross = theta_dml_split_no_cross - alpha)
sd_dml_split_no_cross = sd(theta_dml_split_no_cross)
g_dml_split_no_cross = ggplot(df_dml_split_no_cross, aes(x = theta_dml_split_no_cross)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,9)) +
            stat_function(fun = dnorm, args = list(mean = 0, sd = sd_dml_split_no_cross), geom = "area", col = "red", fill = "red", alpha = 0.01) +
            ggtitle(paste0("No cross-fitting, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml_split_no_cross

ggsave(filename = paste0("simresults/examples_fail/double_ml_resc_no_cross_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_dml_split_no_cross)


df_dml = data.frame(theta_dml = theta_dml - alpha)
sd_dml = sd(theta_dml)
g_dml = ggplot(df_dml, aes(x = theta_dml)) +
            geom_histogram(aes(y=..density..), bins = 100, fill = "dark blue", alpha = 0.3, color = "dark blue") +
            geom_vline(aes(xintercept = 0), col = "black") +
            xlim(c(-0.5, 0.5)) + xlab("") + ylab("") + theme_minimal() + ylim(c(0,9)) +
            stat_function(fun = dnorm, args = list(mean = 0, sd = sd_dml), geom = "area", col = "red", fill = "red", alpha = 0.01) +
            ggtitle(paste0("Cross-fitting, n = ", n_obs, ", p = ", n_vars)) + theme(plot.title = element_text(face="bold", hjust = 0.5))

g_dml

g_comb3 = plot_grid(g_dml_split_no_cross, g_dml)

ggsave(filename = paste0("simresults/examples_fail/nocrossfit_doubleml_n", n_obs, "_p", n_vars, ".pdf"),
        plot = g_comb3, width=8, height=3, dpi=150)

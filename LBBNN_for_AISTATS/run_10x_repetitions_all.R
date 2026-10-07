library(LBBNN)
library(torch)
library(dplyr)

source("train_lbbnn_tau.R")

opt_configs <- read.csv("optimal_configs.csv")

run_single_experiment <- function(x_train, x_test, y_train, y_test, arch_sizes, init_type, tau_val, seed) {
  set.seed(seed)
  torch_manual_seed(seed)
  
  x_mean <- colMeans(x_train); x_sd <- apply(x_train, 2, sd); x_sd[x_sd == 0] <- 1
  x_train <- scale(x_train, center = x_mean, scale = x_sd)
  x_test  <- scale(x_test,  center = x_mean, scale = x_sd)
  
  y_mean <- mean(y_train); y_sd_val <- sd(y_train)
  y_train_sc <- (y_train - y_mean) / y_sd_val
  y_test_sc  <- (y_test  - y_mean) / y_sd_val
  
  tx_train <- torch_tensor(x_train, dtype = torch_float())
  ty_train <- torch_tensor(y_train_sc, dtype = torch_float())
  tx_test  <- torch_tensor(x_test, dtype = torch_float())
  ty_test  <- torch_tensor(y_test_sc, dtype = torch_float())
  
  train_dl <- dataloader(tensor_dataset(tx_train, ty_train), batch_size = length(y_train_sc), shuffle = TRUE)
  test_dl  <- dataloader(tensor_dataset(tx_test, ty_test), batch_size = length(y_test_sc), shuffle = FALSE)
  
  n_layers <- length(arch_sizes) - 1
  alpha <- rep(0.1, n_layers)
  stds  <- 1.0 / sqrt(arch_sizes[-length(arch_sizes)])
  
  if (init_type == "balanced") {
    inits_mat <- matrix(rep(c(-0.5, 0.5), n_layers), nrow = 2)
  } else {
    inits_mat <- matrix(rep(c(-1.5, 1.5), n_layers), nrow = 2)
  }
  
  model <- lbbnn_net(
    problem_type = "regression", sizes = arch_sizes, prior = alpha, std = stds,
    inclusion_inits = inits_mat, input_skip = FALSE, flow = FALSE,
    bias_inclusion_prob = FALSE, custom_act = torch::nn_tanh(), device = "cpu"
  )
  
  capture.output({
    train_lbbnn_tau(epochs = 2000, LBBNN = model, lr = 0.01, train_dl = train_dl, tau = tau_val)
  })
  
  pred <- predict(model, test_dl, num_samples = 200)
  pred_mean <- colMeans(as.array(pred)[,,1])
  rmse <- sqrt(mean((pred_mean - as.numeric(ty_test))^2)) * y_sd_val
  dens <- as.numeric(model$density())
  
  return(c(RMSE = rmse, Density = dens))
}

results <- data.frame()
reps <- 10

cat("Starting 10x repetitions for ALL configs...\n")

# Load real datasets once
boston <- read.csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
xb <- as.matrix(boston[, -14]); yb <- as.numeric(boston[, 14])

yacht <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data", header = FALSE)
xy <- as.matrix(yacht[, 1:6]); yy <- as.numeric(yacht[, 7])

energy <- readxl::read_excel("../../energy.xlsx")
xe <- as.matrix(energy[, 1:8]); ye <- as.numeric(unlist(energy[, 9]))

concrete <- readxl::read_excel("../../concrete.xls")
xc <- as.matrix(concrete[, 1:8]); yc <- as.numeric(unlist(concrete[, 9]))

for (row_idx in 1:nrow(opt_configs)) {
  conf <- opt_configs[row_idx, ]
  ds <- conf$Dataset
  arch <- conf$Arch
  init <- conf$Init
  opt_tau <- conf$Tau
  true_tau <- conf$True_Tau
  
  cat(sprintf("Running %s (%s, %s)\n", ds, arch, init))
  
  for (i in 1:reps) {
    if (ds == "Hernandez") {
      set.seed(i)
      x_tr <- runif(20, -4, 4); y_tr <- x_tr^3 + rnorm(20, 0, 3)
      x_te <- seq(-4, 4, length.out = 200); y_te <- x_te^3 + rnorm(200, 0, 3)
      sizes <- if (arch == "small") c(1, 50, 1) else c(1, 50, 50, 50, 1)
      
      res_opt <- run_single_experiment(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, sizes, init, opt_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "Optimal", RMSE = res_opt[1], Density = res_opt[2]))
      
      res_true <- run_single_experiment(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, sizes, init, true_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "True Variance", RMSE = res_true[1], Density = res_true[2]))
      
    } else if (ds == "Multiscale") {
      set.seed(i)
      x_tr <- runif(80, -3, 3); y_tr <- sin(0.5 * x_tr) + 0.3 * sin(4 * x_tr) + rnorm(80, 0, 0.1)
      x_te <- seq(-3, 3, length.out = 200); y_te <- sin(0.5 * x_te) + 0.3 * sin(4 * x_te) + rnorm(200, 0, 0.1)
      sizes <- if (arch == "small") c(1, 50, 1) else c(1, 50, 50, 50, 1)
      
      res_opt <- run_single_experiment(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, sizes, init, opt_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "Optimal", RMSE = res_opt[1], Density = res_opt[2]))
      
      res_true <- run_single_experiment(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, sizes, init, true_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "True Variance", RMSE = res_true[1], Density = res_true[2]))
      
    } else if (ds == "Gap") {
      set.seed(i)
      x_tr <- c(runif(30, -3, -1.5), runif(30, 1.5, 3)); y_tr <- sin(1.5 * x_tr) + 0.3 * x_tr + rnorm(60, 0, 0.1)
      x_te <- seq(-3, 3, length.out = 200); y_te <- sin(1.5 * x_te) + 0.3 * x_te + rnorm(200, 0, 0.1)
      sizes <- if (arch == "small") c(1, 50, 1) else c(1, 50, 50, 50, 1)
      
      res_opt <- run_single_experiment(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, sizes, init, opt_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "Optimal", RMSE = res_opt[1], Density = res_opt[2]))
      
      res_true <- run_single_experiment(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, sizes, init, true_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "True Variance", RMSE = res_true[1], Density = res_true[2]))
      
    } else if (ds == "Sparse Linear") {
      set.seed(i)
      X <- matrix(rnorm(200 * 20), nrow = 200, ncol = 20)
      beta_true <- rep(0, 20); for(j in 1:5) beta_true[j] <- 1.5*(0.5^(j-1))*(-1)^(j-1)
      y <- X %*% beta_true + rnorm(200, 0, 1.0)
      X_test <- matrix(rnorm(1000 * 20), nrow = 1000, ncol = 20); y_test <- X_test %*% beta_true + rnorm(1000, 0, 1.0)
      sizes <- if (arch == "small") c(20, 50, 1) else c(20, 50, 50, 50, 1)
      
      res_opt <- run_single_experiment(X, X_test, as.numeric(y), as.numeric(y_test), sizes, init, opt_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "Optimal", RMSE = res_opt[1], Density = res_opt[2]))
      
      res_true <- run_single_experiment(X, X_test, as.numeric(y), as.numeric(y_test), sizes, init, true_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "True Variance", RMSE = res_true[1], Density = res_true[2]))
      
    } else if (ds == "Boston") {
      set.seed(i); idx <- sample(1:nrow(xb), 0.8 * nrow(xb))
      sizes <- if (arch == "small") c(ncol(xb), 50, 1) else c(ncol(xb), 50, 50, 50, 1)
      res <- run_single_experiment(xb[idx,], xb[-idx,], yb[idx], yb[-idx], sizes, init, opt_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "Optimal", RMSE = res[1], Density = res[2]))
      
    } else if (ds == "Yacht") {
      set.seed(i); idx <- sample(1:nrow(xy), 0.8 * nrow(xy))
      sizes <- if (arch == "small") c(ncol(xy), 50, 1) else c(ncol(xy), 50, 50, 50, 1)
      res <- run_single_experiment(xy[idx,], xy[-idx,], yy[idx], yy[-idx], sizes, init, opt_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "Optimal", RMSE = res[1], Density = res[2]))
      
    } else if (ds == "Energy") {
      set.seed(i); idx <- sample(1:nrow(xe), 0.8 * nrow(xe))
      sizes <- if (arch == "small") c(ncol(xe), 50, 1) else c(ncol(xe), 50, 50, 50, 1)
      res <- run_single_experiment(xe[idx,], xe[-idx,], ye[idx], ye[-idx], sizes, init, opt_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "Optimal", RMSE = res[1], Density = res[2]))
      
    } else if (ds == "Concrete") {
      set.seed(i); idx <- sample(1:nrow(xc), 0.8 * nrow(xc))
      sizes <- if (arch == "small") c(ncol(xc), 50, 1) else c(ncol(xc), 50, 50, 50, 1)
      res <- run_single_experiment(xc[idx,], xc[-idx,], yc[idx], yc[-idx], sizes, init, opt_tau, i)
      results <- rbind(results, data.frame(Dataset = ds, Arch = arch, Init = init, Type = "Optimal", RMSE = res[1], Density = res[2]))
    }
  }
}

write.csv(results, "final_10x_results_all.csv", row.names = FALSE)
cat("All 10x repetitions finished and saved to final_10x_results_all.csv.\n")

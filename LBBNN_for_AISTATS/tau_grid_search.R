library(LBBNN)
library(torch)
library(readxl)
source("train_lbbnn_tau.R")

run_tau_sweep <- function(x_train, x_test, y_train, y_test, name, is_toy = FALSE, true_scaled_var = NULL) {
  
  # X normalization
  x_mean <- colMeans(x_train)
  x_sd   <- apply(x_train, 2, sd); x_sd[x_sd == 0] <- 1
  x_train <- scale(x_train, center = x_mean, scale = x_sd)
  x_test  <- scale(x_test,  center = x_mean, scale = x_sd)
  
  # y normalization
  y_mean     <- mean(y_train)
  y_sd_val   <- sd(y_train)
  y_train_sc <- (y_train - y_mean) / y_sd_val
  y_test_sc  <- (y_test  - y_mean) / y_sd_val
  
  cat(sprintf("\n=== %s ===\n", name))
  
  tx_train <- torch_tensor(x_train,     dtype = torch_float())
  ty_train <- torch_tensor(y_train_sc,  dtype = torch_float())
  tx_test  <- torch_tensor(x_test,      dtype = torch_float())
  ty_test  <- torch_tensor(y_test_sc,   dtype = torch_float())
  
  bs <- length(y_train_sc)
  train_dl <- dataloader(tensor_dataset(tx_train, ty_train), batch_size = bs, shuffle = TRUE)
  test_dl  <- dataloader(tensor_dataset(tx_test,  ty_test), batch_size = length(y_test_sc), shuffle = FALSE)
  
  P <- ncol(x_train)
  archs <- list(
    small = c(P, 50, 1),
    large = c(P, 50, 50, 50, 1)
  )
  
  if (!is_toy) {
    df <- data.frame(y = y_train_sc, x_train)
    fit <- lm(y ~ ., data = df)
    base_var <- var(fit$residuals)
    cat(sprintf("Base variance (OLS lm estimate): %.6f\n", base_var))
  } else {
    base_var <- true_scaled_var
    cat(sprintf("Base variance (True scaled variance): %.6f\n", base_var))
  }
  
  scales <- c(1.0, 0.5, 0.2, 0.1, 0.05, 0.01)
  tau_grid <- base_var * scales
  
  inits <- c("balanced", "polarized")
  
  cat(sprintf("%-10s %-10s %-10s %-10s %-10s %-10s\n", "Arch", "Init", "Scale", "Tau(sigma2)", "RMSE", "Density"))
  cat("----------------------------------------------------------------------\n")
  
  for (arch_name in names(archs)) {
    sizes <- archs[[arch_name]]
    n_layers <- length(sizes) - 1
    alpha <- rep(0.1, n_layers)
    stds  <- 1.0 / sqrt(sizes[-length(sizes)])
    
    for (init_type in inits) {
      if (init_type == "balanced") {
        inits_mat <- matrix(rep(c(-0.5, 0.5), n_layers), nrow = 2)
      } else {
        inits_mat <- matrix(rep(c(-1.5, 1.5), n_layers), nrow = 2)
      }
      
      for (i in seq_along(tau_grid)) {
        tau_val <- tau_grid[i]
        scale_val <- scales[i]
        
        torch_manual_seed(1)
        model <- lbbnn_net(
          problem_type    = "regression",
          sizes           = sizes,
          prior           = alpha,
          std             = stds,
          inclusion_inits = inits_mat,
          input_skip      = FALSE,
          flow            = FALSE,
          bias_inclusion_prob = FALSE,
          custom_act      = torch::nn_tanh(),
          device          = "cpu"
        )
        
        # We pass tau directly to train_lbbnn_tau (which replaces sigma2)
        capture.output({
          train_lbbnn_tau(epochs = 2000, LBBNN = model, lr = 0.01, train_dl = train_dl, tau = tau_val)
        })
        
        pred      <- predict(model, test_dl, num_samples = 200)
        pred_mean <- colMeans(as.array(pred)[,,1])
        
        rmse <- sqrt(mean((pred_mean - as.numeric(ty_test))^2)) * y_sd_val
        dens <- as.numeric(model$density())
        
        cat(sprintf("%-10s %-10s %-10.2f %-10.6f %-10.4f %-10.4f\n", arch_name, init_type, scale_val, tau_val, rmse, dens))
      }
    }
  }
}

# --- 1. Toy Datasets ---
# Hernandez
set.seed(1)
x_tr <- runif(20, -4, 4); y_tr <- x_tr^3 + rnorm(20, 0, 3)
x_te <- seq(-4, 4, length.out = 200); y_te <- x_te^3 + rnorm(200, 0, 3)
y_sd_h <- sd(y_tr)
run_tau_sweep(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, "Hernandez", is_toy=TRUE, true_scaled_var=9.0/(y_sd_h^2))

# Multiscale
set.seed(1)
x_tr <- runif(80, -3, 3); y_tr <- sin(0.5 * x_tr) + 0.3 * sin(4 * x_tr) + rnorm(80, 0, 0.1)
x_te <- seq(-3, 3, length.out = 200); y_te <- sin(0.5 * x_te) + 0.3 * sin(4 * x_te) + rnorm(200, 0, 0.1)
y_sd_m <- sd(y_tr)
run_tau_sweep(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, "Multiscale", is_toy=TRUE, true_scaled_var=0.01/(y_sd_m^2))

# Gap
set.seed(1)
x_tr <- c(runif(30, -3, -1.5), runif(30, 1.5, 3)); y_tr <- sin(1.5 * x_tr) + 0.3 * x_tr + rnorm(60, 0, 0.1)
x_te <- seq(-3, 3, length.out = 200); y_te <- sin(1.5 * x_te) + 0.3 * x_te + rnorm(200, 0, 0.1)
y_sd_g <- sd(y_tr)
run_tau_sweep(matrix(x_tr, ncol=1), matrix(x_te, ncol=1), y_tr, y_te, "Gap", is_toy=TRUE, true_scaled_var=0.01/(y_sd_g^2))

# Sparse Linear
set.seed(1)
X <- matrix(rnorm(200 * 20), nrow = 200, ncol = 20)
beta_true <- rep(0, 20)
for (j in 1:5) beta_true[j] <- 1.5 * (0.5^(j - 1)) * (-1)^(j - 1)
y <- X %*% beta_true + rnorm(200, 0, 1.0)
X_test <- matrix(rnorm(1000 * 20), nrow = 1000, ncol = 20)
y_test <- X_test %*% beta_true + rnorm(1000, 0, 1.0)
y_sd_sl <- sd(y)
run_tau_sweep(X, X_test, as.numeric(y), as.numeric(y_test), "Sparse Linear", is_toy=TRUE, true_scaled_var=1.0/(y_sd_sl^2))

# --- 2. Real Datasets ---
# Boston
boston <- read.csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
xb <- as.matrix(boston[, -14]); yb <- as.numeric(boston[, 14])
set.seed(1)
idx <- sample(1:nrow(xb), 0.8 * nrow(xb))
run_tau_sweep(xb[idx,], xb[-idx,], yb[idx], yb[-idx], "Boston")

# Yacht
yacht <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data", header = FALSE)
xy <- as.matrix(yacht[, 1:6]); yy <- as.numeric(yacht[, 7])
set.seed(1)
idx <- sample(1:nrow(xy), 0.8 * nrow(xy))
run_tau_sweep(xy[idx,], xy[-idx,], yy[idx], yy[-idx], "Yacht")

# Energy
energy <- read_excel("../../energy.xlsx")
xe <- as.matrix(energy[, 1:8]); ye <- energy[[9]]
set.seed(1)
idx <- sample(1:nrow(xe), 0.8 * nrow(xe))
run_tau_sweep(xe[idx,], xe[-idx,], ye[idx], ye[-idx], "Energy")

# Concrete
concrete <- read_excel("../../concrete.xls")
xc <- as.matrix(concrete[, 1:8]); yc <- concrete[[9]]
set.seed(1)
idx <- sample(1:nrow(xc), 0.8 * nrow(xc))
run_tau_sweep(xc[idx,], xc[-idx,], yc[idx], yc[-idx], "Concrete")

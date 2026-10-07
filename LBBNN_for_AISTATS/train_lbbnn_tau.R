train_lbbnn_tau <- function(epochs, LBBNN, lr, train_dl,
                                 tau  = 1.0,
                                 device  = "cpu",
                                 scheduler     = NULL,
                                 sch_step_size = NULL) {

  opt     <- torch::optim_adam(LBBNN$parameters, lr = lr)
  losses  <- c()
  density <- c()

  if (!is.null(scheduler) && scheduler == "step") {
    sl <- torch::lr_step(opt, step_size = sch_step_size, gamma = 0.1)
  }

  LBBNN$elapsed_time <- 0
  start <- base::proc.time()

  for (epoch in 1:epochs) {
    if (epoch == epochs) { LBBNN$y <- c(); LBBNN$r <- c() }
    LBBNN$train()
    train_loss <- c()

    coro::loop(for (b in train_dl) {
      opt$zero_grad()
      data   <- b[[1]]$to(device = device)
      output <- LBBNN(data, MPM = FALSE)$squeeze()
      target <- b[[2]]$to(device = device)$squeeze()

      if (epoch == epochs) {
        LBBNN$y <- c(LBBNN$y, as.numeric(target$clone()$detach()$cpu()))
        LBBNN$r <- c(LBBNN$r, as.numeric(output$clone()$detach()$cpu()))
      }

      # KL tempering: multiply KL by tau to correct for variational collapse explosion
      loss <- LBBNN$loss_fn(output, target) +
              tau * LBBNN$kl_div() / length(train_dl)

      train_loss <- c(train_loss, loss$item())
      loss$backward()
      opt$step()
    })

    if (!is.null(scheduler) && scheduler == "step") sl$step()

    message(sprintf("\nEpoch %d, training: loss = %3.5f, density = %3.5f [tau=%.3f]\n",
                    epoch, mean(train_loss), LBBNN$density(), tau))
    losses  <- c(losses,  mean(train_loss))
    density <- c(density, LBBNN$density())
  }

  time <- base::proc.time() - start
  LBBNN$elapsed_time <- time[[3]]
  invisible(list(loss = losses, density = density))
}

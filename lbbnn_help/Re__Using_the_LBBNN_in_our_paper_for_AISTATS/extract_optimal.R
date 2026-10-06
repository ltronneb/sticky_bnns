lines <- readLines("scripts_versions/v3_tau_grid_search/grid_summary.txt")

dataset_name <- ""
results <- data.frame()
base_vars <- list()

for (i in 1:length(lines)) {
  l <- lines[i]
  if (grepl("^=== ", l)) {
    dataset_name <- gsub("=== (.*) ===", "\\1", l)
  } else if (grepl("^(small|large)", l)) {
    parts <- unlist(strsplit(l, "\\s+"))
    arch <- parts[1]
    init <- parts[2]
    scale <- as.numeric(parts[3])
    tau <- as.numeric(parts[4])
    rmse <- as.numeric(parts[5])
    dens <- as.numeric(parts[6])
    
    # Store true variance (scale == 1.0)
    if (scale == 1.0) {
      base_vars[[dataset_name]] <- tau
    }
    
    results <- rbind(results, data.frame(
      Dataset = dataset_name, Arch = arch, Init = init, Scale = scale, Tau = tau, RMSE = rmse, Density = dens
    ))
  }
}

# Find optimal per dataset and arch
library(dplyr)
opt <- results %>%
  group_by(Dataset, Arch) %>%
  slice_min(RMSE, n = 1, with_ties = FALSE) %>%
  ungroup()

# Add True_Tau back to it
opt$True_Tau <- sapply(opt$Dataset, function(d) base_vars[[d]])

write.csv(opt, "scripts_versions/v3_tau_grid_search/optimal_configs.csv", row.names = FALSE)
cat("optimal_configs.csv created successfully.\n")

library(parallel)
library(readr)
library(doParallel)
library(foreach)
library(dataverse)
Sys.setenv("DATAVERSE_SERVER" = "dataverse.harvard.edu")

jop <- read_csv("jop_datasets.csv")

fin_dat <- data.frame(dataset = character(0), file = character(0), package_list = character(0))

# Set the number of cores to use
num_cores <- detectCores()

# Register the parallel backend
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Export necessary functions to the worker processes
clusterExport(cl, c("get_dataset", "get_file_by_name"))

# Define a function to process each dataset
process_dataset <- function(i) {
  dataset_id <- sub("^https://doi.org/", "", jop$persistentUrl[i])
  
  dat <- get_dataset(jop$persistentUrl[i])
  r_files <- grep("\\.R$", dat$files$filename, value = TRUE)
  
  if (length(r_files) > 0) {
    results <- foreach(j = 1:length(r_files), .combine = rbind) %do% {
      raw_file <- get_file_by_name(
        filename = r_files[j],
        dataset = dataset_id,
        server = "dataverse.harvard.edu"
      )
      
      writeBin(raw_file, "temp.R")
      text_file <- readLines(file("temp.R", encoding = "UTF-8"))
      
      # Regular expression pattern to match library() and require() commands
      pattern <- "\\b(?:library|require)\\(([^)]+)\\)"
      
      # Extract package names from library() and require() commands
      packages <- unique(gsub(pattern, "\\1", regmatches(text_file, gregexpr(pattern, text_file, perl = TRUE))[[1]]))
      
      # Remove quotes from package names
      packages <- gsub("\"", "", packages)
      
      # Remove arguments such as quietly = TRUE from package names
      packages <- gsub(",\\s*quietly\\s*=\\s*TRUE", "", packages, ignore.case = TRUE)
      
      # Return the row
      data.frame(dataset = jop$persistentUrl[i], file = r_files[j], package_list = toString(packages))
    }
    
    # Combine the results into the fin_dat data frame
    fin_dat <<- rbind(fin_dat, results)
  }
}

# Use foreach to apply the process_dataset function to each dataset
foreach(i = 1:10) %do% {
  process_dataset(i)
}

# Stop the cluster
stopCluster(cl)

# Reset the row names of fin_dat
rownames(fin_dat) <- NULL


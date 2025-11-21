# Find packages imported in R files

library(dataverse)
library(readr)
library(progress)

Sys.setenv("DATAVERSE_SERVER" = "dataverse.harvard.edu")

jop <- read_csv("jop_datasets.csv")

fin_dat <- data.frame(dataset = character(0), file = character(0), package_list = character(0))

pb <- progress_bar$new(total = nrow(jop))

for (i in 1:nrow(jop)) {
  pb$tick()  # Update the progress bar

  dataset_id <- sub("^https://doi.org/", "", jop$persistentUrl[i])

  tryCatch(
    {
      dat <- get_dataset(jop$persistentUrl[i])
      r_files <- grep("\\.R$", dat$files$filename, value = TRUE)

      if (length(r_files) > 0) {
        for (j in 1:length(r_files)) {
          raw_file <- get_file_by_name(
            filename = r_files[j],
            dataset = dataset_id,
            server = "dataverse.harvard.edu"
          )

          writeBin(raw_file, "temp.R")
          text_file <- readLines("temp.R", encoding = "latin1") # encoding = "UTF-8"

          # Regular expression pattern to match library() and require() commands
          pattern <- "\\b(?:library|require)\\(([^)]+)\\)"

          # Extract package names from library() and require() commands
          packages <- unique(gsub(pattern, "\\1", regmatches(text_file, gregexpr(pattern, text_file, perl = TRUE))[[1]]))

          # Remove quotes from package names
          packages <- gsub("\"", "", packages)

          # Remove arguments such as quietly = TRUE from package names
          packages <- gsub(",\\s*quietly\\s*=\\s*TRUE", "", packages, ignore.case = TRUE)

          # Append the row to the fin_dat data frame
          row <- data.frame(dataset = jop$persistentUrl[i], file = r_files[j], package_list = toString(packages))
          fin_dat <- rbind(fin_dat, row)
        }
      }
    },
    error = function(e) {
      # Print the error message
      message("Error processing dataset: ", jop$persistentUrl[i], ". Error message: ", conditionMessage(e))
    }
  )
}

write.csv(fin_dat, "data/data_file_package_imports_2.csv", row.names = FALSE)

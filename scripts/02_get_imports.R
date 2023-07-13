# Find packages imported in R files

library(dataverse)
library(readr)
library(progress)

jop <- read_csv("jop_datasets.csv")

fin_dat <- data.frame(dataset = 1:nrow(jop), file = NA, package_list = NA)

pb <- progress_bar$new(total = nrow(jop))

for(i in 1:(nrow(jop))){
	pb$tick()  # Update the progress bar

	fin_dat$dataset[i] <- jop$persistentUrl[i]
	dataset_id = sub("^https://doi.org/", "", jop$persistentUrl[i])

	dat <- get_dataset(jop$persistentUrl[i])
	r_files <- grep("\\.R$", dat$files$filename, value = TRUE)

	if (length(r_files) > 0) {
		for(j in 1:length(r_files)){
			
			raw_file <- get_file_by_name(
				  filename = r_files[j],
				  dataset  = dataset_id,
				  server   = "dataverse.harvard.edu"
			)

			writeBin(raw_file, "temp.R")
			text_file <- readLines("temp.R", encoding = "UTF-8")

			# Regular expression pattern to match library() and require() commands
			pattern <- "\\b(?:library|require)\\(([^)]+)\\)"

			# Extract package names from library() and require() commands
      		packages <- unique(gsub(pattern, "\\1", regmatches(txt_file, gregexpr(pattern, txt_file, perl = TRUE))[[1]]))

      		# Remove quotes from package names
      		packages <- gsub("\"", "", packages)

      		# Remove arguments such as quietly = TRUE from package names
     		packages <- gsub(",\\s*quietly\\s*=\\s*TRUE", "", packages, ignore.case = TRUE)
     
		  	# Append the row to the fin_dat data frame
	      	row <- data.frame(dataset = jop$persistentUrl[i], file = r_files[j], package_list = toString(packages))
	      	fin_dat <- rbind(fin_dat, row)
		}
	}
}

write.csv(fin_dat, "data_file_package_imports.csv", row.names = FALSE)


# Create an empty vector to store the package names
package_names <- c()

# Parse each string in fin_dat$package_list and extract package names
for (i in 1:length(fin_dat$package_list)) {
  packages <- unlist(strsplit(fin_dat$package_list[i], ",\\s*"))
  packages <- gsub("\"|'", "", packages)
  package_names <- c(package_names, packages)
}

# Create a table of package frequencies
package_table <- table(package_names)

# Convert the table to a data frame and sort by frequency in descending order
package_df <- data.frame(package_name = names(package_table), frequency = as.numeric(package_table))
package_df <- package_df[order(-package_df$frequency), ]

# Print the resulting table
print(package_df)

}

### Dataset level

# Create a new data frame to store the reshaped data
reshaped_dat <- data.frame(dataset = unique(fin_dat$dataset), package_list = NA)

# Iterate over each dataset
for (i in 1:nrow(reshaped_dat)) {
  # Get the dataset URL
  dataset_url <- reshaped_dat$dataset[i]
  
  # Subset fin_dat for the current dataset
  subset_dat <- fin_dat[fin_dat$dataset == dataset_url, ]
  
  # Get the unique package names for the current dataset
  unique_packages <- unique(unlist(strsplit(subset_dat$package_list, ",\\s*")))
  unique_packages <- gsub("\"|'", "", unique_packages)
  
  # Store the unique package names in the reshaped data frame
  reshaped_dat$package_list[i] <- toString(unique_packages)
}




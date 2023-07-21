# Reshape, Tabulate, Plot

library(readr)
library(tidyverse)
library(knitr)

fin_dat_1 <- read_csv("data/data_file_package_imports_1.csv")
fin_dat_2 <- read_csv("data/data_file_package_imports_2.csv")
fin_dat <- rbind(fin_dat_1, fin_dat_2)

# Single quotes
fin_dat$package_list <- gsub("'", "", fin_dat$package_list)

# Separate the package_list into individual rows
separated_data <- fin_dat %>%
  filter(!is.na(fin_dat$package_list)) %>%
  separate_rows(package_list, sep = ", ")

# Count the frequency of each package
package_frequency <- separated_data %>%
  count(package_list, sort = TRUE)

# Reverse sort the frequencies
package_frequency <- arrange(package_frequency, desc(n))

# View the resulting dataframe
package_frequency

# Group at dataset level
grouped_dat <- fin_dat %>%
  filter(!is.na(package_list)) %>%
  group_by(dataset) %>%
  summarize(package_list = paste(unique(package_list), collapse = ", "))

# Separate the package_list into individual rows
separated_data <- grouped_data %>%
  separate_rows(package_list, sep = ", ")

# Count the frequency of each package
package_frequency <- separated_data %>%
  count(package_list, sort = TRUE)

# Reverse sort the frequencies
package_frequency <- arrange(package_frequency, desc(n))

# View the resulting dataframe
package_frequency
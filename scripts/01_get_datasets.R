library(httr)
library(jsonlite)

# Set the base URL of your Dataverse instance
base_url <- "https://dataverse.harvard.edu/api"

# Set the Dataverse alias or identifier
dataverse_id <- "polbehavior"

# Construct the API endpoint URL
url <- paste0(base_url, "/dataverses/", dataverse_id, "/contents")

# Make a GET request to retrieve the dataset information
response <- GET(url)

# Extract the content from the response
content <- content(response, "text")

# Parse the JSON content
dataset_json <- fromJSON(content)

# Convert the dataset information to a data frame
df <- as.data.frame(dataset_json$data)

# Write the data frame to a CSV file
write.csv(df, "polbehavior_datasets.csv", row.names = FALSE)

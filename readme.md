### Softverse: Auto-compute Citations to Software From Replication Files

We analyze replication files from 34 social science journals including the APSR, AJPS, JoP, BJPolS, Political Analysis, World Politics, Political Behavior, etc. posted to the Harvard Dataverse to tally the libraries used. This can be used as a way to calculate citation metrics for software.

see: https://gojiberries.io/2023/07/02/hard-problems-about-research-software/

### Scripts

1. [Datasets by Dataverse](scripts/01_get_datasets_for_dataverses.ipynb) produces [list of datasets by dataverse (.gz)](data/datasets_by_dataverse.tar.gz)

2. [List And Download All (R) Scripts Per Dataset](scripts/02_get_scripts_per_dataset.ipynb) takes the files from step #1 and produces [list of files per dataset (.gz)](data/02_get_scripts_per_dataset.ipynb) and downloads those scripts (dump [here](data/script_files.tar.gz))

3. [Regex the files to tally imports](scripts/03_tally_imports.ipynb) takes the output from step #2 and produces [imports per file](data/file_imports.csv) and [imports per package](data/imports_per_package.csv) (if there are multiple imports per repository, we only count it once). A snippet of that last file can be seen below.

p.s. Deprecated R Files [here](scripts/r/)

#### Top R Package Imports

| package       | count |
|---------------|-------|
| ggplot2       | 1322  |
| foreign       | 1009  |
| stargazer     | 901   |
| dplyr         | 789   |
| tidyverse     | 720   |
| xtable        | 608   |
| plyr          | 485   |
| lmtest        | 451   |
| MASS          | 442   |
| gridExtra     | 420   |
| sandwich      | 394   |
| haven         | 356   |
| car           | 342   |
| readstata13   | 339   |
| reshape2      | 324   |
| stringr       | 318   |
| texreg        | 273   |
| data.table    | 263   |
| scales        | 257   |
| tidyr         | 253   |
| grid          | 247   |
| lme4          | 241   |
| Hmisc         | 236   |
| lubridate     | 223   |
| readxl        | 218   |
| broom         | 195   |
| lfe           | 190   |
| RColorBrewer  | 188   |
| ggpubr        | 188   |
| estimatr      | 174   |


### Authors

Gaurav Sood and Daniel Weitzel

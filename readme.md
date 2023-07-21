### Softverse: Auto-compute Citations to Software From Replication Files

We analyze replication files of articles in the APSR, AJPS, JoP, BJPolS, Political Analysis, World Politics, and Political Behavior posted to the Harvard Dataverse to tally the libraries used. This can be used as a way to calculate citation metrics for software.

see: https://gojiberries.io/2023/07/02/hard-problems-about-research-software/


### Scripts

1. [Datasets by Dataverse](scripts/01_get_datasets_for_dataverses.ipynb) produces [list of datasets by dataverse (.gz)](data/datasets_by_dataverse.gz)

2. [(.R) Scripts For Scripts Per Dataset](scripts/02_get_scripts_per_dataset.ipynb) takes the files from step #1 and produces [list of files per dataset (.gz)](data/02_get_scripts_per_dataset.ipynb) and downloads those scripts (dump [here](data/script_files.gz))

3. [Regex the files to tally imports](scripts/03_tally_imports.ipynb) takes the output from step #2 and produces [imports per file](data/file_imports.csv) and [imports per package](data/imports_per_package.csv). A snippet of that last file can be seen below.

p.s. Deprecated R Files [here](scripts/r/)

#### Top R Package Imports

| imports     |   count |
|:------------|--------:|
| ggplot2     |     249 |
| dplyr       |     229 |
| stargazer   |     146 |
| foreign     |     144 |
| data.table  |      80 |
| xtable      |      79 |
| readstata13 |      79 |
| lmtest      |      76 |
| plyr        |      72 |
| tidyverse   |      70 |
| AER         |      68 |
| pacman      |      67 |
| gridExtra   |      54 |
| stringr     |      49 |
| sandwich    |      47 |
| tidyr       |      44 |
| MASS        |      43 |
| lfe         |      41 |
| reshape2    |      39 |
| ivpack      |      38 |



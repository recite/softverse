### Softverse: Auto-compute Citations to Software From Replication Files

We iterate over replication files in the JoP dataverse to provide a PoC for how we can auto-compute citations to software from replication files.

see: https://gojiberries.io/2023/07/02/hard-problems-about-research-software/

1. [Datasets in (JoP) Dataverse](scripts/01_get_datasets.R)
2. [Package imports in R files](scripts/02_get_imports.R)
3. [Plot](scripts/03_plot.R)

#### Top R Package Imports in JoP

|package_list |  n|
|:------------|--:|
|plyr         | 28|
|magrittr     | 26|
|foreign      | 21|
|tidyverse    | 12|
|ggplot2      | 11|
|lmtest       |  9|
|rstan        |  9|
|haven        |  7|
|stargazer    |  7|



### Softverse: Auto-compute Citations to Software From Replication Files

[![CI](https://github.com/recite/softverse/actions/workflows/ci.yml/badge.svg)](https://github.com/recite/softverse/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/softverse.svg)](https://pypi.python.org/pypi/softverse)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://recite.github.io/softverse/)
[![PePy Downloads](https://static.pepy.tech/badge/softverse)](https://www.pepy.tech/projects/softverse)

We analyze replication files from 34 social science journals including the APSR, AJPS, JoP, BJPolS, Political Analysis, World Politics, Political Behavior, etc. posted to the Harvard Dataverse to tally the libraries used. This can be used as a way to calculate citation metrics for software.

see: https://gojiberries.io/2023/07/02/hard-problems-about-research-software/

## Installation

### Prerequisites
- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Softverse
```bash
uv pip install softverse
```

### Development Setup
```bash
git clone https://github.com/recite/softverse.git
cd softverse
uv sync --all-extras
```

## Usage

### Quick Start

Run the complete data collection and analysis pipeline:

```bash
uv run run-pipeline
```

### Individual Components

#### 1. Collect Datasets from Dataverse
```bash
# Collect datasets using configuration file
uv run collect-datasets --config config/settings.yaml --output-dir outputs/data/datasets/

# Force refresh to re-download all datasets
uv run collect-datasets --force-refresh --output-dir outputs/data/datasets/

# Use custom CSV input file with dataverse information
uv run collect-datasets --input-csv data/dataverse_socialscience.csv --output-dir outputs/data/datasets/
```

#### 2. Collect Scripts and Code Files
```bash
# Collect scripts from all sources (Dataverse, Zenodo, ICPSR)
uv run collect-scripts --source all --base-output-dir outputs/scripts/

# Collect only from Dataverse
uv run collect-scripts --source dataverse --datasets-dir outputs/data/datasets/ --base-output-dir outputs/scripts/

# Collect from Zenodo with specific communities
uv run collect-scripts --source zenodo --zenodo-communities harvard-dataverse --max-zenodo-records 1000

# Collect from ICPSR with query
uv run collect-scripts --source icpsr --icpsr-query "political science" --max-icpsr-studies 500
```

#### 3. Analyze Software Imports
```bash
# Analyze imports from collected scripts
uv run analyze-imports --scripts-dir outputs/scripts/ --output-dir outputs/analysis/

# Specify script patterns to analyze
uv run analyze-imports --scripts-dir outputs/scripts/ --output-dir outputs/analysis/ --config config/settings.yaml
```

#### 4. Collect from OSF (Open Science Framework)
```bash
# Note: OSF collector is available via Python API (see below)
from softverse.collectors import OSFCollector
```

#### 5. Collect from ResearchBox
```bash
# Note: ResearchBox collector is available via Python API (see below)
from softverse.collectors import ResearchBoxCollector
```

### Configuration

#### API Keys and Authentication

API keys can be configured in two ways:

1. **Environment Variables** (Recommended for security):
```bash
export DATAVERSE_API_KEY="your-dataverse-api-key"
export OSF_API_TOKEN="your-osf-personal-access-token"
export ZENODO_ACCESS_TOKEN="your-zenodo-access-token"
export ICPSR_USERNAME="your-icpsr-username"
export ICPSR_PASSWORD="your-icpsr-password"
```

2. **Configuration File** (`config/settings.yaml`):
```yaml
dataverse:
  base_url: "https://dataverse.harvard.edu"
  api_key: "your-api-key-here"  # Optional, for authenticated requests
  input_csv: "data/dataverse_socialscience.csv"

osf:
  api_token: "your-osf-token-here"  # Optional, for authenticated requests
  rate_limit_delay: 0.5

zenodo:
  access_token: "your-zenodo-token"  # Optional, for authenticated requests
  communities: ["harvard-dataverse"]

icpsr:
  username: "your-username"
  password: "your-password"

researchbox:
  base_url: "https://researchbox.org"
  concurrent_downloads: 5

# Output directories
output:
  datasets_dir: "outputs/data/datasets"
  scripts_dir: "outputs/scripts"
  analysis_dir: "outputs/analysis"
  logs_dir: "outputs/logs"
```

**Note:** Environment variables take precedence over configuration file values for sensitive data like API keys.

### Python API

```python
from pathlib import Path
from softverse.collectors import (
    DataverseCollector,
    OSFCollector,
    ResearchBoxCollector,
    ZenodoCollector
)
from softverse.analyzers import ImportAnalyzer

# Initialize collectors
dataverse = DataverseCollector()
osf = OSFCollector()
researchbox = ResearchBoxCollector()

# Collect datasets from Harvard Dataverse
datasets = dataverse.collect_from_dataverse_csv(
    csv_path="data/dataverse_socialscience.csv",
    output_dir=Path("outputs/datasets")
)

# Search and collect from OSF
osf_results = osf.search_nodes("reproducibility")
osf.collect_nodes(
    node_ids=[node["id"] for node in osf_results[:10]],
    output_dir=Path("outputs/osf"),
    download_files=True
)

# Collect from ResearchBox
researchbox.collect_range(
    start_id=1,
    end_id=100,
    output_dir=Path("outputs/researchbox"),
    extract=True
)

# Analyze R package imports
analyzer = ImportAnalyzer()
results = analyzer.analyze_directory(
    directory=Path("outputs/scripts"),
    output_dir=Path("outputs/analysis")
)

# Get summary statistics
summary = analyzer.generate_summary_statistics(results)
print(f"Total scripts analyzed: {summary['total_files']}")
print(f"Top R packages: {summary['top_r_packages'][:10]}")
print(f"Top Python packages: {summary['top_python_packages'][:10]}")
```

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

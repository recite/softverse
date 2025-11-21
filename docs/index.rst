Softverse Documentation
========================

**Softverse** is a Python package for auto-computing citations to software from replication files.

It collects script files from research repositories (Dataverse, Zenodo, ICPSR) and analyzes
software package usage to generate citation statistics.

Features
--------

* **Multi-source Collection**: Collects scripts from Harvard Dataverse, Zenodo, and ICPSR
* **Storage Optimized**: Efficient archive processing with minimal storage requirements
* **Incremental Processing**: Supports resumable operations with checkpoint system
* **Parallel Processing**: Configurable workers for optimal performance
* **Error Recovery**: Robust error handling with automatic retry logic
* **Comprehensive CLI**: Full command-line interface for automation and CRON jobs

Quick Start
-----------

Installation::

    pip install softverse

Basic usage::

    # Collect datasets from Dataverse
    collect-datasets

    # Collect scripts from all sources
    collect-scripts --source all

    # Analyze software imports
    analyze-imports

    # Run complete pipeline
    run-pipeline

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   cli/index
   configuration
   examples

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API:

   api/collectors
   api/analyzers
   api/utils

CLI Reference
=============

.. toctree::
   :maxdepth: 1
   :caption: CLI:

   cli/collect-datasets
   cli/collect-scripts
   cli/analyze-imports
   cli/run-pipeline

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

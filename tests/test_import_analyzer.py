"""Test import analyzer functionality."""

import json

import pytest

from softverse.analyzers.import_analyzer import ImportAnalyzer


class MockAnalyzer:
    """Mock analyzer for testing without config dependencies."""

    def __init__(self):
        self.analyzer = ImportAnalyzer.__new__(ImportAnalyzer)
        # Initialize only what we need for testing

    def __getattr__(self, name):
        return getattr(self.analyzer, name)


@pytest.fixture
def analyzer():
    """Create a mock ImportAnalyzer instance for testing."""
    return MockAnalyzer()


class TestPythonImportExtraction:
    """Test Python import extraction."""

    def test_basic_imports(self, analyzer):
        """Test basic Python import extraction."""
        python_code = """
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
"""
        imports = analyzer.extract_python_imports_from_code(python_code)
        expected = {
            "os",
            "sys",
            "pathlib",
            "collections",
            "numpy",
            "sklearn",
            "sklearn.model_selection",
        }
        assert set(imports) == expected

    def test_notebook_imports(self, analyzer):
        """Test Jupyter notebook import extraction."""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["import pandas as pd\n", "import numpy as np"],
                },
                {"cell_type": "markdown", "source": ["# This is markdown"]},
                {
                    "cell_type": "code",
                    "source": [
                        "from sklearn import metrics\n",
                        "import matplotlib.pyplot as plt",
                    ],
                },
            ]
        }
        notebook_json = json.dumps(notebook_content)
        imports = analyzer.extract_python_imports_from_notebook(notebook_json)
        expected = {"pandas", "numpy", "sklearn", "matplotlib", "matplotlib.pyplot"}
        assert set(imports) == expected


class TestRImportExtraction:
    """Test R import extraction."""

    def test_basic_r_imports(self, analyzer):
        """Test basic R import extraction."""
        r_code = """
library(dplyr)
require(ggplot2)
library("readr")
require('tidyr')
"""
        imports = analyzer.extract_r_imports(r_code)
        expected = {"dplyr", "ggplot2", "readr", "tidyr"}
        assert set(imports) == expected

    def test_pacman_imports(self, analyzer):
        """Test pacman p_load extraction."""
        r_code = """
p_load(dplyr, ggplot2, readr)
p_load("tidyr", "stringr")
"""
        imports = analyzer.extract_r_imports(r_code)
        expected = {"dplyr", "ggplot2", "readr", "tidyr", "stringr"}
        assert set(imports) == expected

    def test_namespace_operators(self, analyzer):
        """Test namespace operator extraction."""
        r_code = """
dplyr::filter(data, condition)
ggplot2::ggplot() + ggplot2::geom_point()
base::mean(x)  # This should be excluded
"""
        imports = analyzer.extract_r_imports(r_code)
        # base should be excluded as it's a base package
        assert "dplyr" in imports
        assert "ggplot2" in imports
        assert "base" not in imports

    def test_conditional_loading(self, analyzer):
        """Test conditional loading patterns."""
        r_code = """
if (!require(pacman)) install.packages("pacman")
if(!require("devtools")) install.packages("devtools")
"""
        imports = analyzer.extract_r_imports(r_code)
        expected = {"pacman", "devtools"}
        assert set(imports) == expected


class TestStataImportExtraction:
    """Test Stata import extraction."""

    def test_ssc_install(self, analyzer):
        """Test ssc install pattern."""
        stata_code = """
ssc install outreg2
ssc install estout, replace
"""
        imports = analyzer.extract_stata_imports(stata_code)
        expected = {"outreg2", "estout"}
        assert set(imports) == expected

    def test_net_install(self, analyzer):
        """Test net install pattern."""
        stata_code = """
net install rdrobust, from(https://raw.githubusercontent.com/rdpackages/rdrobust/master/stata) replace
net install ivreg2
"""
        imports = analyzer.extract_stata_imports(stata_code)
        expected = {"rdrobust", "ivreg2"}
        assert set(imports) == expected

    def test_command_usage(self, analyzer):
        """Test known command usage detection."""
        stata_code = """
estout using results.tex
outreg2 using table1.doc
coefplot, drop(_cons)
"""
        imports = analyzer.extract_stata_imports(stata_code)
        expected = {"estout", "outreg2", "coefplot"}
        assert set(imports) == expected

    def test_which_command(self, analyzer):
        """Test which command pattern."""
        stata_code = """
which ivreg2
which custom_command
which summarize  // This should be excluded as built-in
"""
        imports = analyzer.extract_stata_imports(stata_code)
        assert "ivreg2" in imports
        assert "custom_command" in imports
        assert "summarize" not in imports  # Built-in command


class TestCommentHandling:
    """Test comment handling across languages."""

    def test_r_comments(self, analyzer):
        """Test R comment handling."""
        r_code = """
library(dplyr)  # Load dplyr
# library(excluded)  # This should be ignored
library(ggplot2)
"""
        imports = analyzer.extract_r_imports(r_code)
        expected = {"dplyr", "ggplot2"}
        assert set(imports) == expected
        assert "excluded" not in imports

    def test_stata_comments(self, analyzer):
        """Test Stata comment handling."""
        stata_code = """
ssc install outreg2  // Install outreg2
/* ssc install excluded */  // This should be ignored
ssc install estout
"""
        imports = analyzer.extract_stata_imports(stata_code)
        expected = {"outreg2", "estout"}
        assert set(imports) == expected
        assert "excluded" not in imports


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_python_syntax(self, analyzer):
        """Test handling of invalid Python syntax."""
        invalid_code = "import os\nthis is not valid python syntax"
        imports = analyzer.extract_python_imports_from_code(invalid_code)
        # With invalid syntax, AST parsing fails and returns empty list
        # This is expected behavior
        assert imports == []

    def test_invalid_notebook_json(self, analyzer):
        """Test handling of invalid notebook JSON."""
        invalid_json = "not valid json"
        imports = analyzer.extract_python_imports_from_notebook(invalid_json)
        assert imports == []

    def test_empty_content(self, analyzer):
        """Test handling of empty content."""
        assert analyzer.extract_python_imports_from_code("") == []
        assert analyzer.extract_r_imports("") == []
        assert analyzer.extract_stata_imports("") == []

    def test_only_comments(self, analyzer):
        """Test handling of files with only comments."""
        r_comments = "# This is just a comment\n# Another comment"
        assert analyzer.extract_r_imports(r_comments) == []

        stata_comments = "// Just comments\n/* Block comment */"
        assert analyzer.extract_stata_imports(stata_comments) == []


if __name__ == "__main__":
    pytest.main([__file__])

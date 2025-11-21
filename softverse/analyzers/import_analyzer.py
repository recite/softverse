"""Import analysis for script files."""

import ast
import json
import re
import warnings
from pathlib import Path

import pandas as pd

try:
    from rpy2.robjects import r

    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False

from softverse.config import get_config
from softverse.utils.file_utils import (
    CheckpointManager,
    ensure_directory,
    get_file_list,
)
from softverse.utils.logging_utils import LogProgress, get_logger

warnings.filterwarnings("ignore")
logger = get_logger("import_analyzer")


class ImportAnalyzer:
    """Analyzes import statements in script files."""

    def __init__(self, config_path: str | None = None):
        """Initialize analyzer.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.checkpoint_manager = CheckpointManager()

    def extract_r_imports_ast(self, text: str) -> list[str]:
        """Extract R imports using AST parsing (requires rpy2).

        Args:
            text: R script content

        Returns:
            List of package names from AST analysis
        """
        if not R_AVAILABLE:
            return []

        try:
            # Parse R code into AST
            r_code = f"""
            library(utils)
            parsed <- try(parse(text = {repr(text)}), silent = TRUE)
            if (inherits(parsed, "try-error")) {{
                list()
            }} else {{
                # Extract function calls from parsed expressions
                calls <- list()
                extract_calls <- function(expr) {{
                    if (is.call(expr)) {{
                        calls <<- c(calls, list(expr))
                        for (i in seq_along(expr)[-1]) {{
                            extract_calls(expr[[i]])
                        }}
                    }} else if (is.list(expr)) {{
                        for (e in expr) extract_calls(e)
                    }}
                }}
                for (expr in parsed) extract_calls(expr)
                # Find library/require calls
                packages <- character(0)
                for (call in calls) {{
                    if (is.call(call) && length(call) >= 2) {{
                        func_name <- as.character(call[[1]])
                        if (func_name %in% c("library", "require", "loadNamespace", "requireNamespace")) {{
                            if (is.character(call[[2]])) {{
                                packages <- c(packages, call[[2]])
                            }} else if (is.name(call[[2]])) {{
                                packages <- c(packages, as.character(call[[2]]))
                            }}
                        }}
                    }}
                }}
                unique(packages)
            }}
            """

            result = r(r_code)
            return [str(pkg) for pkg in result if pkg and str(pkg) != "NULL"]

        except Exception as e:
            logger.debug(f"R AST parsing failed, falling back to regex: {e}")
            return []

    def extract_r_imports(self, text: str) -> list[str]:
        """Extract R library imports from text using both AST and pattern matching.

        Args:
            text: R script content

        Returns:
            List of imported library names
        """
        imports = set()

        # Try AST approach first if available
        if R_AVAILABLE:
            try:
                ast_imports = self.extract_r_imports_ast(text)
                imports.update(ast_imports)
            except Exception as e:
                logger.debug(f"R AST parsing failed: {e}")

        # Always do pattern matching as fallback/supplement
        # Remove comments to avoid false matches
        text_lines = []
        for line in text.split("\n"):
            # Remove comments (but preserve strings)
            if "#" in line:
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i - 1] != "\\"):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif char == "#" and not in_string:
                        line = line[:i]
                        break
            text_lines.append(line)

        clean_text = " ".join(text_lines)

        # Pattern 1: library() and require() calls
        pattern1 = (
            r"\b(?:library|require)\s*\(\s*(?:[\"']([^\"']+)[\"']|([^,\)\s]+))\s*[,\)]"
        )
        for match in re.finditer(pattern1, clean_text, flags=re.IGNORECASE):
            pkg = match.group(1) or match.group(2)
            if pkg and pkg.strip():
                imports.add(pkg.strip())

        # Pattern 2: p_load() from pacman
        pattern2 = r"\bp_load\s*\(\s*([^\)]+)\)"
        for match in re.finditer(pattern2, clean_text, flags=re.IGNORECASE):
            # Split by comma and clean up
            packages = match.group(1).split(",")
            for pkg in packages:
                pkg_clean = re.sub(r"[\"\s]", "", pkg).strip()
                if pkg_clean and not pkg_clean.startswith("char"):
                    imports.add(pkg_clean)

        # Pattern 3: Namespace operators (pkg::function)
        pattern3 = r"\b([a-zA-Z][a-zA-Z0-9._]*)\s*::"
        for match in re.finditer(pattern3, clean_text):
            pkg = match.group(1)
            # Exclude common non-package prefixes
            if pkg not in ["base", "utils", "stats", "graphics", "datasets"]:
                imports.add(pkg)

        # Pattern 4: loadNamespace() and requireNamespace()
        pattern4 = (
            r"\b(?:loadNamespace|requireNamespace)\s*\(\s*[\"']([^\"']+)[\"']\s*\)"
        )
        for match in re.finditer(pattern4, clean_text, flags=re.IGNORECASE):
            imports.add(match.group(1))

        # Pattern 5: Conditional loading (if (!require(...)))
        pattern5 = r"!\s*require\s*\(\s*(?:[\"']([^\"']+)[\"']|([^,\)\s]+))\s*[,\)]"
        for match in re.finditer(pattern5, clean_text, flags=re.IGNORECASE):
            pkg = match.group(1) or match.group(2)
            if pkg and pkg.strip():
                imports.add(pkg.strip())

        # Pattern 6: devtools::install_github and similar
        pattern6 = r"\binstall_github\s*\(\s*[\"']([^/\"']+)/([^\"']+)[\"']"
        for match in re.finditer(pattern6, clean_text, flags=re.IGNORECASE):
            imports.add(match.group(2))  # Package name from user/package format

        # Clean up imports
        cleaned_imports = []
        for imp in imports:
            # Remove version specifications and options
            imp_clean = re.sub(r"\s*[>=<].*", "", imp)
            imp_clean = re.sub(r"\s*\(.*\)", "", imp_clean)
            imp_clean = imp_clean.strip()

            # Skip empty, numeric, or obviously invalid package names
            if (
                imp_clean
                and not imp_clean.isdigit()
                and re.match(r"^[a-zA-Z][a-zA-Z0-9._]*$", imp_clean)
                and len(imp_clean) > 1
            ):
                cleaned_imports.append(imp_clean)

        return list(set(cleaned_imports))

    def classify_python_import(self, module_name: str) -> str:
        """Classify Python import as standard library, third-party, or local.

        Args:
            module_name: Python module name

        Returns:
            Classification: 'standard', 'third-party', or 'local'
        """
        import importlib.util
        import sys

        # Get top-level module name
        top_level = module_name.split(".")[0]

        try:
            # Try to find the module spec
            spec = importlib.util.find_spec(top_level)

            if spec is None:
                # Module not found, likely local or invalid
                return "local"

            if spec.origin is None:
                # Built-in module
                return "standard"

            # Check if it's in the standard library by looking at the path
            stdlib_path = Path(sys.prefix)
            if spec.origin:
                try:
                    origin_path = Path(spec.origin)
                    # Check if module is in stdlib path (Python 3.9+ has is_relative_to)
                    if hasattr(origin_path, "is_relative_to"):
                        if origin_path.is_relative_to(stdlib_path):
                            return "standard"
                    else:
                        # Fallback for older Python versions
                        if str(stdlib_path) in str(origin_path):
                            return "standard"
                except (OSError, ValueError):
                    pass

            # Check for site-packages (third-party)
            if spec.origin and "site-packages" in str(spec.origin):
                return "third-party"

            # Check for local modules (relative imports or current directory)
            if top_level.startswith("_") or (
                spec.origin and spec.origin.startswith(".")
            ):
                return "local"

            # Default fallback
            return "third-party"

        except (ImportError, ModuleNotFoundError, AttributeError):
            # Fallback to simple heuristics if import fails
            if top_level.startswith("_"):
                return "local"

            # Common third-party packages
            common_third_party = {
                "numpy",
                "pandas",
                "matplotlib",
                "scipy",
                "sklearn",
                "requests",
                "flask",
                "django",
                "pytest",
                "click",
                "tqdm",
                "pillow",
                "opencv",
                "tensorflow",
                "torch",
                "keras",
                "seaborn",
                "plotly",
                "streamlit",
            }

            if top_level.lower() in common_third_party:
                return "third-party"

            return "third-party"  # Default assumption

    def classify_r_import(self, package_name: str) -> str:
        """Classify R import as base, recommended, or third-party.

        Args:
            package_name: R package name

        Returns:
            Classification: 'base', 'recommended', or 'third-party'
        """
        # R base packages
        base_packages = {
            "base",
            "compiler",
            "datasets",
            "graphics",
            "grDevices",
            "grid",
            "methods",
            "parallel",
            "splines",
            "stats",
            "stats4",
            "tcltk",
            "tools",
            "translations",
            "utils",
        }

        # R recommended packages
        recommended_packages = {
            "boot",
            "class",
            "cluster",
            "codetools",
            "foreign",
            "KernSmooth",
            "lattice",
            "mass",
            "Matrix",
            "mgcv",
            "nlme",
            "nnet",
            "rpart",
            "spatial",
            "survival",
        }

        package_lower = package_name.lower()
        if package_lower in base_packages:
            return "base"
        elif package_lower in recommended_packages:
            return "recommended"
        else:
            return "third-party"

    def extract_python_imports_from_notebook(self, notebook_content: str) -> list[str]:
        """Extract Python imports from Jupyter notebook JSON.

        Args:
            notebook_content: JSON content of .ipynb file

        Returns:
            List of imported module names
        """
        imports = []

        try:
            notebook = json.loads(notebook_content)

            # Extract code from all code cells
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])
                    # Join source lines into single string
                    if isinstance(source, list):
                        code = "".join(source)
                    else:
                        code = str(source)

                    # Extract imports from this cell
                    cell_imports = self.extract_python_imports_from_code(code)
                    imports.extend(cell_imports)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error parsing Jupyter notebook: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error parsing notebook: {e}")

        return list(set(imports))

    def extract_python_imports_from_code(self, text: str) -> list[str]:
        """Extract Python imports from code text using AST.

        Args:
            text: Python code content

        Returns:
            List of imported module names
        """
        imports = []

        try:
            tree = ast.parse(text)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name:
                            # Get top-level module name
                            top_level = alias.name.split(".")[0]
                            imports.append(top_level)
                            # Also add full module path for submodules
                            if "." in alias.name:
                                imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get top-level module name
                        top_level = node.module.split(".")[0]
                        imports.append(top_level)
                        # Also add full module path for submodules
                        if "." in node.module:
                            imports.append(node.module)

        except SyntaxError as e:
            logger.debug(f"Python syntax error (expected for some files): {e}")
        except Exception as e:
            logger.warning(f"Error parsing Python imports: {e}")

        return list(set(imports))

    def extract_python_imports(self, text: str, is_notebook: bool = False) -> list[str]:
        """Extract Python imports from text.

        Args:
            text: Python script content or notebook JSON
            is_notebook: Whether the content is a Jupyter notebook

        Returns:
            List of imported module names
        """
        if is_notebook:
            return self.extract_python_imports_from_notebook(text)
        else:
            return self.extract_python_imports_from_code(text)

    def extract_stata_imports(self, text: str) -> list[str]:
        """Extract Stata package/command imports from text.

        Args:
            text: Stata script content

        Returns:
            List of imported package/command names
        """
        imports = set()

        # Remove comments (/* */ and // style)
        # Remove /* */ comments first
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        # Remove // comments line by line
        text_lines = []
        for line in text.split("\n"):
            if "//" in line:
                line = line.split("//")[0]
            text_lines.append(line)

        clean_text = " ".join(text_lines)

        # Pattern 1: ssc install
        pattern1 = r"\bssc\s+install\s+([a-zA-Z][a-zA-Z0-9_]*)"
        for match in re.finditer(pattern1, clean_text, flags=re.IGNORECASE):
            imports.add(match.group(1))

        # Pattern 2: net install
        pattern2 = r"\bnet\s+install\s+([a-zA-Z][a-zA-Z0-9_]*)"
        for match in re.finditer(pattern2, clean_text, flags=re.IGNORECASE):
            imports.add(match.group(1))

        # Pattern 3: adopath (path management for ado files)
        pattern3 = r"\badopath\s+\+\s+([^\s;]+)"
        for match in re.finditer(pattern3, clean_text, flags=re.IGNORECASE):
            # Extract directory name which often indicates package name
            path = match.group(1).strip("\"'")
            if "/" in path or "\\" in path:
                pkg_name = path.split("/")[-1].split("\\")[-1]
                if pkg_name and re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", pkg_name):
                    imports.add(pkg_name)

        # Pattern 4: which command (checking if command exists - indicates package usage)
        pattern4 = r"\bwhich\s+([a-zA-Z][a-zA-Z0-9_]*)"
        for match in re.finditer(pattern4, clean_text, flags=re.IGNORECASE):
            cmd = match.group(1)
            # Exclude built-in Stata commands
            stata_builtins = {
                "summarize",
                "regress",
                "logit",
                "probit",
                "tabulate",
                "describe",
                "generate",
                "replace",
                "drop",
                "keep",
                "merge",
                "append",
                "sort",
                "list",
                "display",
                "use",
                "save",
                "clear",
                "set",
                "quietly",
                "capture",
                "foreach",
                "forvalues",
                "if",
                "else",
                "while",
            }
            if cmd.lower() not in stata_builtins:
                imports.add(cmd)

        # Pattern 5: Common user-written commands that suggest packages
        # Look for commands that are likely from packages (not built-in)
        pattern5 = r"\b((?:est|outreg|outreg2|estout|esttab|coefplot|binscatter|rd|rdrobust|ivreg2|xtivreg2|xtabond2|did|diff|matching|psmatch2|ttest|ranksum|signrank)(?:2?)?)\b"
        for match in re.finditer(pattern5, clean_text, flags=re.IGNORECASE):
            cmd = match.group(1)
            # Map known commands to their package names
            package_map = {
                "outreg": "outreg",
                "outreg2": "outreg2",
                "estout": "estout",
                "esttab": "estout",
                "coefplot": "coefplot",
                "binscatter": "binscatter",
                "rdrobust": "rdrobust",
                "ivreg2": "ivreg2",
                "xtivreg2": "ivreg2",
                "xtabond2": "xtabond2",
                "psmatch2": "psmatch2",
            }
            pkg_name = package_map.get(cmd.lower(), cmd)
            imports.add(pkg_name)

        # Pattern 6: findit command (searching for packages)
        pattern6 = r"\bfindit\s+([a-zA-Z][a-zA-Z0-9_]*)"
        for match in re.finditer(pattern6, clean_text, flags=re.IGNORECASE):
            imports.add(match.group(1))

        # Clean up imports
        cleaned_imports = []
        for imp in imports:
            imp_clean = imp.strip()
            if (
                imp_clean
                and re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", imp_clean)
                and len(imp_clean) > 1
            ):
                cleaned_imports.append(imp_clean)

        return list(set(cleaned_imports))

    def extract_imports_from_file(self, file_path: Path) -> pd.DataFrame | None:
        """Extract imports from a single file.

        Args:
            file_path: Path to script file

        Returns:
            DataFrame with import information or None if no imports
        """
        try:
            # Read file with fallback encoding
            try:
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, encoding="latin1") as f:
                    text = f.read()

            # Determine language and extract imports
            suffix = file_path.suffix.lower()
            imports = []
            language = None

            if suffix in [".R", ".r", ".rscript"]:
                imports = self.extract_r_imports(text)
                language = "R"
            elif suffix == ".py":
                imports = self.extract_python_imports(text, is_notebook=False)
                language = "Python"
            elif suffix == ".ipynb":
                imports = self.extract_python_imports(text, is_notebook=True)
                language = "Python"
            elif suffix in [".do"]:
                imports = self.extract_stata_imports(text)
                language = "Stata"

            if imports and language:
                df = pd.DataFrame(
                    {
                        "package": imports,
                        "language": [language] * len(imports),
                        "file_path": [str(file_path)] * len(imports),
                        "classification": [
                            (
                                self.classify_python_import(pkg)
                                if language == "Python"
                                else (
                                    self.classify_r_import(pkg)
                                    if language == "R"
                                    else "third-party"
                                )
                            )  # Default for Stata
                            for pkg in imports
                        ],
                    }
                )
                return df

        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")

        return None

    def extract_imports_from_directory(
        self, directory: Path, extensions: list[str] | None = None
    ) -> pd.DataFrame:
        """Extract imports from all files in a directory.

        Args:
            directory: Directory to search
            extensions: File extensions to process

        Returns:
            DataFrame with all import information
        """
        if extensions is None:
            extensions = self.config.get(
                "processing.valid_extensions", [".R", ".py", ".do", ".ipynb"]
            )

        # Get all files
        files = get_file_list(directory, extensions)
        logger.info(f"Found {len(files)} files to analyze in {directory}")

        all_imports = []
        processed_count = 0

        with LogProgress(logger, "extracting imports", len(files)) as progress:
            for file_path in files:
                try:
                    imports_df = self.extract_imports_from_file(file_path)
                    if imports_df is not None and not imports_df.empty:
                        all_imports.append(imports_df)
                        processed_count += 1

                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")

                progress.update()

        if all_imports:
            result_df = pd.concat(all_imports, ignore_index=True)
            logger.info(
                f"Extracted {len(result_df)} imports from {processed_count} files"
            )
            return result_df
        else:
            logger.warning("No imports found in directory")
            return pd.DataFrame(
                columns=["package", "language", "file_path", "classification"]
            )

    def add_repository_info(self, imports_df: pd.DataFrame) -> pd.DataFrame:
        """Add repository information to imports DataFrame.

        Args:
            imports_df: DataFrame with import information

        Returns:
            DataFrame with added repository information
        """
        if imports_df.empty:
            return imports_df

        logger.info("Adding repository information to imports")

        # Extract dataverse and repo_id from file paths
        # Expected format: scripts/{source}_datasets_files/{repo_id}/...
        imports_df = imports_df.copy()

        # Extract dataverse/source information
        imports_df["dataverse"] = imports_df["file_path"].str.extract(
            r"/([^/]+)_datasets_files/"
        )
        imports_df["repo_id"] = imports_df["file_path"].str.extract(
            r"_datasets_files/([^/]+)/"
        )

        # For other sources (Zenodo, ICPSR), extract differently
        zenodo_mask = imports_df["file_path"].str.contains("zenodo", na=False)
        if zenodo_mask.any():
            imports_df.loc[zenodo_mask, "dataverse"] = "zenodo"
            imports_df.loc[zenodo_mask, "repo_id"] = imports_df.loc[
                zenodo_mask, "file_path"
            ].str.extract(r"zenodo/([^/]+)/")

        icpsr_mask = imports_df["file_path"].str.contains("icpsr", na=False)
        if icpsr_mask.any():
            imports_df.loc[icpsr_mask, "dataverse"] = "icpsr"
            imports_df.loc[icpsr_mask, "repo_id"] = imports_df.loc[
                icpsr_mask, "file_path"
            ].str.extract(r"icpsr/([^/]+)/")

        return imports_df

    def create_package_summary(self, imports_df: pd.DataFrame) -> pd.DataFrame:
        """Create package-level summary with citation counts.

        Args:
            imports_df: DataFrame with import information

        Returns:
            DataFrame with package citation counts
        """
        if imports_df.empty:
            return pd.DataFrame(columns=["package", "count"])

        logger.info("Creating package citation summary")

        # Group by repository to count each package only once per repo
        repo_level = (
            imports_df.groupby(["dataverse", "repo_id"])["package"]
            .agg(set)
            .reset_index()
        )

        # Explode sets to individual package entries
        exploded = repo_level.explode("package")

        # Count citations per package
        citation_counts = (
            exploded.groupby("package")
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
            .reset_index(drop=True)
        )

        logger.info(f"Found {len(citation_counts)} unique packages")
        return citation_counts

    def analyze_all_scripts(
        self,
        scripts_dir: str | None = None,
        output_dir: str | None = None,
        languages: list[str] | None = None,
        force_refresh: bool = False,
    ) -> bool:
        """Analyze imports from all collected scripts.

        Args:
            scripts_dir: Directory containing script files
            output_dir: Output directory for analysis results
            languages: Languages to analyze (R, Python, Stata)
            force_refresh: Force refresh analysis

        Returns:
            True if analysis successful
        """
        # Get configuration
        config = self.config.output_config
        scripts_dir_path = Path(scripts_dir or "scripts/")
        output_dir_path = Path(output_dir or config.get("base_dir", "data/"))

        ensure_directory(output_dir_path)

        # Determine file extensions based on languages
        if languages is None:
            languages = ["R", "Python", "Stata"]

        extensions = []
        if "R" in languages:
            extensions.extend([".R", ".r", ".Rscript"])
        if "Python" in languages:
            extensions.extend([".py", ".ipynb"])
        if "Stata" in languages:
            extensions.extend([".do"])

        try:
            logger.info(f"Starting import analysis of {scripts_dir_path}")
            logger.info(f"Languages: {', '.join(languages)}")
            logger.info(f"Extensions: {', '.join(extensions)}")

            # Extract imports from all scripts
            imports_df = self.extract_imports_from_directory(
                scripts_dir_path, extensions
            )

            if imports_df.empty:
                logger.warning("No imports found to analyze")
                return False

            # Add repository information
            imports_df = self.add_repository_info(imports_df)

            # Output detailed imports file
            file_imports_path = output_dir_path / config.get(
                "files.file_imports", "file_imports.csv"
            )
            imports_df.to_csv(file_imports_path, index=False)
            logger.info(f"Saved detailed imports to {file_imports_path}")

            # Create package summary
            package_summary = self.create_package_summary(imports_df)

            # Output package summary file
            package_imports_path = output_dir_path / config.get(
                "files.package_imports", "imports_per_package.csv"
            )
            package_summary.to_csv(package_imports_path, index=False)
            logger.info(f"Saved package summary to {package_imports_path}")

            # Print top packages for verification
            logger.info("Top 10 most cited packages:")
            for _, row in package_summary.head(10).iterrows():
                logger.info(f"  {row['package']}: {row['count']} citations")

            # Summary statistics
            total_imports = len(imports_df)
            total_packages = len(package_summary)
            total_repos = imports_df[["dataverse", "repo_id"]].nunique().sum()

            logger.info("Analysis complete:")
            logger.info(f"  Total imports: {total_imports:, }")
            logger.info(f"  Unique packages: {total_packages:, }")
            logger.info(f"  Repositories analyzed: {total_repos:, }")

            return True

        except Exception as e:
            logger.error(f"Import analysis failed: {e}")
            return False


def main():
    """Main entry point for import analysis."""
    import argparse

    from softverse.utils.logging_utils import setup_logging

    parser = argparse.ArgumentParser(description="Analyze imports from script files")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--scripts-dir", type=str, help="Directory containing script files"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for analysis results"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["R", "Python", "Stata"],
        help="Languages to analyze",
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force refresh analysis"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Create analyzer and run
    analyzer = ImportAnalyzer(args.config)
    success = analyzer.analyze_all_scripts(
        scripts_dir=args.scripts_dir,
        output_dir=args.output_dir,
        languages=args.languages,
        force_refresh=args.force_refresh,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()

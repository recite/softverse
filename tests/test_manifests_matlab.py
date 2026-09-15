"""A MATLAB kernel's version is the kernel's, not MATLAB's."""

from __future__ import annotations

import json

from softverse.detect.manifests import read_notebook_environment


def _notebook(name: str, version: str) -> str:
    return json.dumps(
        {"metadata": {"language_info": {"name": name, "version": version}}}
    )


def test_matlab_kernel_version_is_not_reported_as_matlab():
    # MetaKernel's matlab_kernel, as found in all three deposits carrying one.
    assert read_notebook_environment(_notebook("matlab", "0.16.11")) is None


def test_python_kernel_version_is_still_read():
    read = read_notebook_environment(_notebook("python", "3.11.5"))
    assert read is not None
    assert read.signals == {"python_version": "3.11.5"}

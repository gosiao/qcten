"""
Unit and regression test for the qcten package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import qcten


def test_qcten_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "qcten" in sys.modules

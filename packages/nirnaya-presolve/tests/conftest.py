"""
Makes the test-only `nirnaya_core` shim (tests/_core_shim/nirnaya_core)
importable as `nirnaya_core` for the duration of the test suite, standing
in for the real nirnaya-core package until Part 1 is available for
integration (see docs/ASSUMED_CORE_API.md).

A real, installed `nirnaya_core` always takes precedence: the shim path is
only appended to sys.path (not prepended), and only if `nirnaya_core`
isn't already importable. Once real integration happens, delete
tests/_core_shim entirely; this file then becomes a no-op and can be
deleted too.
"""

import importlib.util
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
_SRC_DIR = os.path.abspath(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

if importlib.util.find_spec("nirnaya_core") is None:
    _SHIM_DIR = os.path.join(os.path.dirname(__file__), "_core_shim")
    if _SHIM_DIR not in sys.path:
        sys.path.append(_SHIM_DIR)

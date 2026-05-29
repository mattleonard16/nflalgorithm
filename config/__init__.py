"""Compatibility shim: re-export symbols from the sibling ``config.py`` module.

The repo has both this ``config/`` package (holding example config + JSON
side-files) and a top-level ``config.py`` (the runtime config, gitignored as
proprietary). Python's import resolution gives precedence to the package,
so ``from config import config`` fails because this ``__init__.py`` does not
define ``config``. To preserve every existing ``from config import config``
callsite without restructuring the repo, we load the sibling ``config.py``
file by absolute path and re-export its public attributes here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.py"

if not _CONFIG_FILE.exists():
    raise ImportError(
        f"Expected runtime config at {_CONFIG_FILE}. "
        "Copy config/config.example.py to ./config.py and adjust values."
    )

_spec = importlib.util.spec_from_file_location("_nflalgorithm_runtime_config", _CONFIG_FILE)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_module, _name)

del _name, _spec, _module, _CONFIG_FILE

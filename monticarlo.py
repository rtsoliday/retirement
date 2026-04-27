"""Backward-compatible shim for the Tkinter retirement GUI.

The main module is named ``montecarlo``. Older docs/tests used the
misspelled ``monticarlo`` name, so keep this wrapper for imports and scripts.
"""

from montecarlo import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("montecarlo", run_name="__main__")

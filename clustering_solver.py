"""Compatibility entry point; use python -m connections.clustering instead."""
from connections.clustering import *

if __name__ == "__main__":
    import runpy
    runpy.run_module("connections.clustering", run_name="__main__")

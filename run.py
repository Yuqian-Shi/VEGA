#!/usr/bin/env python3
import os
import sys

# Add src to the Python path so the vega package can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vega.__main__ import main

if __name__ == "__main__":
    main()


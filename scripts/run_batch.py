#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# Ensure project root is on sys.path when launching as: python scripts/run_batch.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_transfer_runner.cli import main

if __name__ == "__main__":
    main()

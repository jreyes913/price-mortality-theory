#!/bin/bash
set -e  # stop on first failure

cd /home/jose/price-mortality-theory

venv/bin/python3 scripts/bulk_sweep.py
venv/bin/python3 scripts/garch_comparison.py
venv/bin/python3 scripts/combo_comparison.py
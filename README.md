# Leakguard — Order-Book ML Research Validation

Leakguard is a research-engineering project for testing whether
short-horizon order-book machine-learning experiments are trustworthy.

It checks data integrity and feature causality, uses purged
chronological validation, compares simple model baselines,
and creates reproducible research reports.

It is not a production HFT engine or proof of profitable alpha.

## Quick Start

### Windows PowerShell

Create a virtual environment:

```powershell
py -m venv .venv

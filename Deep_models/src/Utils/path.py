from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = Path(__file__).resolve().parents[3]

# Data directory - use shared data from Baselines_model
_data_candidates = [
    ROOT_DIR / 'data',
    WORKSPACE_DIR / 'Baselines_model' / 'data',
    WORKSPACE_DIR / 'Full_model' / 'data',
]

DATA_DIR = None
for candidate in _data_candidates:
    if candidate.exists():
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    DATA_DIR = WORKSPACE_DIR / 'Baselines_model' / 'data'

OUTPUT_DIR = ROOT_DIR / 'output'
CACHE_DIR = ROOT_DIR / 'cache'

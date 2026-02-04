from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Primary data directory - try multiple locations
_data_candidates = [
    ROOT_DIR / 'datasets',
    ROOT_DIR / 'data',
    ROOT_DIR.parent / 'Baselines_model' / 'data',
    ROOT_DIR.parent / 'ceemdan_EVloss' / 'data',
]

DATA_DIR = None
for candidate in _data_candidates:
    if candidate.exists():
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    # Default fallback - create datasets folder
    DATA_DIR = ROOT_DIR / 'datasets'

OUTPUT_DIR = ROOT_DIR / 'output'


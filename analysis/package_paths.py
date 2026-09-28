"""Portable paths for frozen-data analysis. Inputs are never output locations."""
from pathlib import Path
import os
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / 'data'
OUTPUT_ROOT = Path(os.environ.get('GEORISK_OUTPUT_DIR', PACKAGE_ROOT / 'outputs')).resolve()
SOURCE_ROOT = OUTPUT_ROOT / 'source'
RUNS_ROOT = DATA_ROOT / 'runs'

def read_tile_scores(path):
    """CSV exports preserve float64 values and original record ordering exactly."""
    return pd.read_csv(Path(path).with_suffix('.csv'), float_precision='round_trip',
                       dtype={'sample_id': str, 'event_id': str})

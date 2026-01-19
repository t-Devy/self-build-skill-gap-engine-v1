import pandas as pd

from pathlib import Path

def load_raw_data(csv_path: Path) -> pd.DataFrame:

    if not isinstance(csv_path, Path):
        raise ValueError("load_raw_data() expects <csv_path>: str")
    return pd.read_csv(csv_path)
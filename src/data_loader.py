"""Loading and basic reshaping of the SKU x Location demand panel."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def load_raw_data(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the long-format SKU x Location CSV (date, sku, location, demand, ...).

    This project uses only the real dataset provided — there is no
    synthetic-data fallback. If the file isn't there yet, run
    `python -m src.prepare_real_data --input <path_to_original_csv>` first
    (see data/README.md).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run "
            "`python -m src.prepare_real_data --input <path_to_original_csv>` "
            "to generate it from the original file, or set SC_DATA_PATH to "
            "point at an already-prepared CSV. See data/README.md."
        )
    df = pd.read_csv(path)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    return df


def get_series(df: pd.DataFrame, sku: str, location: str) -> pd.Series:
    """Return the demand series for a single (sku, location) pair, indexed by date."""
    mask = (df[config.SKU_COL] == sku) & (df[config.LOCATION_COL] == location)
    subset = df.loc[mask].set_index(config.DATE_COL).sort_index()
    return subset[config.TARGET_COL]


def list_sku_location_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return all distinct (sku, location) combinations present in the data."""
    pairs = df[[config.SKU_COL, config.LOCATION_COL]].drop_duplicates()
    return list(pairs.itertuples(index=False, name=None))

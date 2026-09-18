from pathlib import Path

import pandas as pd

from schemas import DatasetRowChurn

DATASET_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def to_dataset_rows(df: pd.DataFrame) -> list[DatasetRowChurn]:
    return [DatasetRowChurn(**row) for row in df.to_dict(orient="records")]

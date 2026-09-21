from dataset import load_dataset


def test_load_dataset_reads_real_csv():
    df = load_dataset()

    assert len(df) > 0
    assert "churn" in df.columns

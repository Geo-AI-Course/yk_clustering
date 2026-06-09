import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "resources/stat_data_mock.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "ms_ezor"]
    assert (df[feature_cols].sum(axis=1).round(6) <= 1.0 + 1e-6).all(), (
        "Some rows have feature values summing to more than 1"
    )
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    feature_cols = [c for c in df.columns if c != "ms_ezor"]

    print(f"Total rows: {len(df)}")
    print(f"Features: {feature_cols}\n")

    train_df, test_df = split_data(df)

    print(f"Train size: {len(train_df)} ({len(train_df)/len(df):.0%})")
    print(f"Test size:  {len(test_df)} ({len(test_df)/len(df):.0%})")
    print(f"\nTrain ms_ezor values: {sorted(train_df['ms_ezor'].tolist())}")
    print(f"Test  ms_ezor values: {sorted(test_df['ms_ezor'].tolist())}")

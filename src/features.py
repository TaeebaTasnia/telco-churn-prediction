import pandas as pd


SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport'
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, float('inf')],
        labels=['0-12', '13-24', '25-48', '49+'],
        include_lowest=True
    )

    df['num_services'] = df[SERVICE_COLS].apply(
        lambda row: (row == 'Yes').sum(), axis=1
    )

    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)

    df['has_streaming'] = (
        (df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')
    ).astype(int)

    return df
